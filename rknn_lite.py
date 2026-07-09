# -*- coding:utf-8 -*-

import os
import copy
import traceback
import platform
from .rknn_platform_utils import get_host_os_platform, get_librknn_api_require_dll_dir, list_support_target_platform
from .rknn_runtime import RKNNRuntime
from .rknn_log import set_log_level_and_file_path
from .npu_config.cpu_npu_mapper import get_support_target_soc


class RKNNLite:

    NPU_CORE_AUTO  = 0                                   # default, run on NPU core randomly.
    NPU_CORE_0     = 1                                   # run on NPU core 0.
    NPU_CORE_1     = 2                                   # run on NPU core 1.
    NPU_CORE_2     = 4                                   # run on NPU core 2.
    NPU_CORE_0_1   = 3                                   # run on NPU core 1 and core 2.
    NPU_CORE_0_1_2 = 7                                   # run on NPU core 1 and core 2 and core 3.
    NPU_CORE_ALL   = 0xffff                              # run on all NPU cores.

    """
    Rockchip NN Kit
    """
    def __init__(self, verbose=False, verbose_file=None):
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if get_host_os_platform() == 'Windows_x64':
            require_dll_dir = get_librknn_api_require_dll_dir()
            new_path = os.environ["PATH"] + ";" + require_dll_dir
            os.environ["PATH"] = new_path
        self.target = 'simulator'
        self.verbose = verbose
        if verbose_file is not None:
            if os.path.dirname(verbose_file) != "" and not os.path.exists(os.path.dirname(verbose_file)):
                verbose_file = None
        self.rknn_log = set_log_level_and_file_path(verbose, verbose_file)

        # get rknn-toolkit-lite2 version
        try:
            import pkg_resources
            self.rknn_log.w('rknn-toolkit-lite2 version: ' +
                            pkg_resources.get_distribution("rknn-toolkit-lite2").version)
        except Exception:
            pass

        if verbose:
            if verbose_file is None:
                self.rknn_log.w('Verbose file path is invalid, debug info will not dump to file.')
            else:
                self.rknn_log.d('Save log info to: {}'.format(verbose_file))
        self.rknn_data = None
        self.load_model_in_npu = False
        self.rknn_runtime = None
        self.root_dir = cur_path

    def load_rknn(self, path):
        """
        Load RKNN model
        :param path: RKNN model file path
        :return: success: 0, failure: -1
        """
        if not os.path.exists(path):
            self.rknn_log.e('Invalid RKNN model path: {}'.format("None" if (path is None or path == "") else path),
                            False)
            return -1
        try:
            # Read RKNN model file data
            with open(path, 'rb') as f:
                self.rknn_data = f.read()
        except:
            self.rknn_log.e('Catch exception when loading RKNN model [{}]!'.format(path), False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        if self.rknn_data is None:
            return -1

        return 0

    def list_devices(self):
        """
        print all adb devices and devices use ntb.
        :return: adb_devices, list; ntb_devices, list. example:
                 adb_devices = ['rk3568']
                 ntb_devices = ['rk3588']
        """
        # get adb devices
        adb_devices = RKNNRuntime.get_adb_devices()
        # get ntb devices
        ntb_devices = RKNNRuntime.get_ntb_devices()
        adb_devices_copy = copy.deepcopy(adb_devices)
        for device in adb_devices_copy:
            if device in ntb_devices:
                adb_devices.remove(device)
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0:
            self.rknn_log.p('all device(s) with adb mode:')
            self.rknn_log.p(",".join(adb_devices))
        if len(ntb_devices) > 0:
            self.rknn_log.p('all device(s) with ntb mode:')
            self.rknn_log.p(",".join(ntb_devices))
        if len(adb_devices) == 0 and len(ntb_devices) == 0:
            self.rknn_log.p('None devices connected.')
        self.rknn_log.p('*' * 25)
        if len(adb_devices) > 0 and len(ntb_devices) > 0:
            all_adb_devices_are_ntb_also = True
            for device in adb_devices:
                if device not in ntb_devices:
                    all_adb_devices_are_ntb_also = False
            if not all_adb_devices_are_ntb_also:
                self.rknn_log.w('Cannot use devices in adb/ntb mode at the same time.')
        return adb_devices, ntb_devices

    def init_runtime(self, target=None, device_id=None, async_mode=False, core_mask=NPU_CORE_AUTO):
        """
        Init run time environment. Needed by called before inference or eval performance.
        :param target: target platform, RK3562/RK3566/Rk3568/RK3588.
                       None means NPU inside.
        :param device_id: adb device id, only needed when multiple devices connected to pc
        :param async_mode: enable or disable async mode
        :param core_mask: npu core mode, default uses auto mode, value is RKNNLite.NPU_CORE_0.
                          RKNNLite.NPU_CORE_AUTO : auto mode, default value.
                          RKNNLite.NPU_CORE_0    : core 0 mode
                          RKNNLite.NPU_CORE_1    : core 1 mode
                          RKNNLite.NPU_CORE_2    : core 2 mode
                          RKNNLite.NPU_CORE_0_1  : combine core 0/1 mode
                          RKNNLite.NPU_CORE_0_1_2: combine core 0/1/2 mode, only supported by RK3588
        :return: success: 0, failure: -1
        """
        self.rknn_log.d('target set by user is: {}'.format(target))

        if target is None and platform.machine() != 'aarch64' and platform.machine() != 'armv7l':
            support_target = 'RK3562 / RK3566 / RK3568 / RK3588'
            self.rknn_log.e("RKNN Toolkit Lite2 does not support simulator, please specify the target: {}", False).\
                format(support_target)
            return -1

        if self.rknn_data is None:
            self.rknn_log.e("Model is not loaded yet, this interface should be called after load_rknn!", False)
            return -1

        # if rknn_runtime is not None, release it first
        if self.rknn_runtime is not None:
            self.rknn_runtime.release()
            self.rknn_runtime = None
        try:
            self.rknn_runtime = RKNNRuntime(root_dir=self.root_dir, target=target, device_id=device_id,
                                            async_mode=async_mode, core_mask=core_mask)
        except:
            self.rknn_log.e('Catch exception when init runtime!', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        # build graph with runtime
        try:
            self.rknn_runtime.build_graph(self.rknn_data, self.load_model_in_npu)
        except:
            self.rknn_log.e('Catch exception when init runtime!', False)
            if target is not None and target.upper() in get_support_target_soc() and platform.machine() != "armv7l":
                adb_devices, ntb_devices = self.list_devices()
                for device in adb_devices:
                    if device in ntb_devices:
                        adb_devices.remove(device)
                devices = adb_devices + ntb_devices
                self.rknn_log.e('{}'.format(devices), False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        # set core mask, only valid for RK3576 / RK3588
        try:
            ret = self.rknn_runtime.set_core_mask(core_mask)
        except:
            self.rknn_log.e('Catch exception when set npu core mode.', False)
            return -1

        # check runtime version
        try:
            self.rknn_runtime.check_rt_version()
        except:
            self.rknn_log.e('Catch exception when init runtime!', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return -1

        # extend runtime library if needed
        self.rknn_runtime.extend_rt_lib()

        # dynamic shape check
        self.rknn_runtime.check_dynamic_shape()

        return ret

    def inference(self, inputs, data_type=None, data_format=None, inputs_pass_through=None, get_frame_id=False):
        """
        Run model inference
        :param inputs: Input data List (ndarray list)
        :param data_type: Data type (str), currently support: int8, uint8, int16, float16, float32, default uint8
        :param data_format: Data format (str), current support: 'nhwc' and 'nchw', default is None.
        :param inputs_pass_through: set pass_through flag(0 or 1: 0 meas False, 1 means True) for every input. (list)
        :param get_frame_id: weather need get output/input frame id when using async mode,it can be use in camera demo
        :return: Output data (ndarray list)
        """
        if self.rknn_runtime is None:
            self.rknn_log.e('Runtime environment is not inited, please call init_runtime to init it first!', False)
            return None

        # set inputs
        try:
            self.rknn_runtime.set_inputs(inputs, data_type, data_format, inputs_pass_through=inputs_pass_through)
        except:
            self.rknn_log.e('Catch exception when setting inputs.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        # run
        try:
            ret = self.rknn_runtime.run(get_frame_id)
        except:
            self.rknn_log.e('Catch exception when running RKNN model.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        # get outputs
        try:
            outputs = self.rknn_runtime.get_outputs(get_frame_id)
        except:
            self.rknn_log.e('Catch exception when getting outputs.', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        if not get_frame_id:
            return outputs
        else:
            outputs.append(ret[1])
            return outputs

    def get_sdk_version(self):
        """
        Get SDK version
        :return: sdk_version
        """
        if self.rknn_runtime is None:
            self.rknn_log.e('Runtime environment is not inited, please call init_runtime to init it first!', False)
            return None

        try:
            sdk_version, _, _ = self.rknn_runtime.get_sdk_version()
        except:
            self.rknn_log.e('Catch exception when get sdk version', False)
            self.rknn_log.e(traceback.format_exc(), False)
            return None

        return sdk_version

    def list_support_target_platform(self, rknn_model=None):
        """
        List all target platforms which can run the model in rknn_model.
        :param rknn_model: RKNN model path, if None, all target platforms will be printed, and ordered by NPU model.
        :return: support_target(dict)
        """
        if rknn_model is not None and not os.path.exists(rknn_model):
            self.rknn_log.e('The model {} does not exist.'.format(rknn_model))
            return None
        return list_support_target_platform(rknn_model)

    def release(self):
        """
        Release RKNN resource
        :return: None
        """
        # release rknn runtime
        if self.rknn_runtime is not None:
            self.rknn_runtime.release()
            self.rknn_runtime = None

