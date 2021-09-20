import copy

from queue import Queue
from model.mix_op import *
from model.proxyless_nets import *


class SuperProxylessNASNets(ProxylessNASNets):

    def __init__(self, mode, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0):
        
        self.mode = mode
        self.width_stages = width_stages
        self.n_cell_stages = n_cell_stages
        self.conv_candidates = conv_candidates
        self.stride_stages = stride_stages
        self.n_classes = n_classes
        self.width_mult = width_mult
        self.bn_param = bn_param
        self.dropout_rate = dropout_rate

        self._redundant_modules = None
        self._unused_modules = None

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')

        # first block
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(['3x3_MBConv1'], input_channel, first_cell_width, 1, 'weight_bn_act'))
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks
        blocks = [first_block]
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(modified_conv_candidates, input_channel, width, stride, 'weight_bn_act'))
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        if mode == 'supervised':
            classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        elif mode == 'SimCLR':
            classifier = MLP(in_features=last_channel, out_features=128, dropout_rate=dropout_rate)
        super(SuperProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def get_clone_net(self):
        return SuperProxylessNASNets(
            self.mode, self.width_stages, self.n_cell_stages, self.conv_candidates, self.stride_stages,
            self.n_classes, self.width_mult, self.bn_param, self.dropout_rate)

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def convert_to_normal_net(self, net):
        net_queue = Queue()
        self_queue = Queue()
        net_queue.put(net)
        self_queue.put(self)

        while not net_queue.empty() and not self_queue.empty():
            net_module = net_queue.get()
            self_module = self_queue.get()
            for (net_m, self_m) in zip(net_module._modules, self_module._modules):
                net_child = net_module._modules[net_m]
                self_child = self_module._modules[self_m]
                if net_child is None:
                    continue
                if net_child.__str__().startswith('MixedEdge'):
                    self_module._modules[self_m] = net_child.chosen_op
                else:
                    net_queue.put(net_child)
                    self_queue.put(self_child)
        return ProxylessNASNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)
