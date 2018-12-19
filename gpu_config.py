os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def gpu_config():
	"""
	gpu配置
	:return: 
	"""
	# tensorflow gpu配置
	gpuConfig = tf.ConfigProto()
	gpuConfig.allow_soft_placement = False  # 设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
	gpuConfig.gpu_options.allow_growth = True  # 设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
	gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.95  # 程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值
	return gpuConfig
