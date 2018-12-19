os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def gpu_config():
	"""
	gpu����
	:return: 
	"""
	# tensorflow gpu����
	gpuConfig = tf.ConfigProto()
	gpuConfig.allow_soft_placement = False  # ����ΪTrue����GPU�����ڻ��߳����г���GPU�������еĴ���ʱ���Զ��л���CPU����
	gpuConfig.gpu_options.allow_growth = True  # ����ΪTrue����������ʱ������ݳ�������GPU�Դ������������С����Դ
	gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.95  # �������е�ʱ�������GPU�Դ���Դ���������rate���趨ֵ
	return gpuConfig
