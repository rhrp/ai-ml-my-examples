	X�����?X�����?!X�����?	I��J�I@I��J�I@!I��J�I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X�����?kH�c�C�?A7T��7��?YD�ͩd��?*	�/�d,�@2U
Iterator::Model::ParallelMapV2��Tގp�?!'�5��8@)��Tގp�?1'�5��8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZ)r�c�?!��o�H@)���_�5�?1�MՂ�r8@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�b�D(�?!�*���5@)�b�D(�?1�*���5@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map�U�&���?!����.5@)P9&����?1�Y�a0@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatqqTn���?!�{��X4@)������?1E�>�@:Preprocessing2F
Iterator::ModelE� y�?!�����:@)˃�9D�?1<��X�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatG�I�ѯ?!����.��?)��z�"0�?1��ԫR��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip 7��?!��B!J@)t��gy�?1�2�C��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�P����?!�*�V��?)����H��?1�%4Q���?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeyͫ:��?!��е��?)yͫ:��?1��е��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev���?!K�^W�u�?)�Ѫ�t��?1b`��C�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	�`8�0C�?!�=�����?)�`8�0C�?1�=�����?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�4}v��?!-Zf�'J�?)K�*nܒ?1�t0ޗ�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate���t�?!e��gAW�?)|~!<�?1�T5u�x�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate�kЗ���?!��Q��?)h�,{�?1�
�>t�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor:vP��h?!�8�ʵ>�?):vP��h?1�8�ʵ>�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor)���^b?!#���A�?))���^b?1#���A�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor ����]?!������?) ����]?1������?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[1]::FromTensor��R�h\?!̌	V��?)��R�h\?1̌	V��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor�/��CX?!Ta'7��?)�/��CX?1Ta'7��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)H4�"V?!U������?))H4�"V?1U������?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice¾�D�Q?!6��khR�?)¾�D�Q?16��khR�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�k���DP?!gy���Y�?)�k���DP?1gy���Y�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�'eRCK?!ۼn�-n�?)�'eRCK?1ۼn�-n�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSlice�N^�E?!hJ[�z�?)�N^�E?1hJ[�z�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 51.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t20.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9H��J�I@I�'�~�vH@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	kH�c�C�?kH�c�C�?!kH�c�C�?      ��!       "      ��!       *      ��!       2	7T��7��?7T��7��?!7T��7��?:      ��!       B      ��!       J	D�ͩd��?D�ͩd��?!D�ͩd��?R      ��!       Z	D�ͩd��?D�ͩd��?!D�ͩd��?b      ��!       JCPU_ONLYYH��J�I@b q�'�~�vH@