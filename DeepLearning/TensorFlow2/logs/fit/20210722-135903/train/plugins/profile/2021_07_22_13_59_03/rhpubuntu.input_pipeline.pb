	DP5z5 �?DP5z5 �?!DP5z5 �?	nD��1�J@nD��1�J@!nD��1�J@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$DP5z5 �?(F�̱�?A�l�?3��?YVc	k��?*�S㥛n�@)      `=2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?;�bF�?!@��?cC8@)?;�bF�?1@��?cC8@:Preprocessing2U
Iterator::Model::ParallelMapV2��M(D��?!��0Q��6@)��M(D��?1��0Q��6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw�ӂ��?!���G@) A�c���?1HX_���4@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?�'I���?!�_�v�8@)��Rb��?1�&��*0@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatY�E����?!�q�N3!@)��_����?1F����@:Preprocessing2F
Iterator::Model�GW���?!sE�*�8@)z��w)u�?1���D� @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�0�q�	�?!%�����?)���?1�o�<tH�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�KqU�7�?!�B_�/+I@)'h��'��?1=�l��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range¥c�3��?!p)�'�?)¥c�3��?1p)�'�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�*P��Ô?!�\]����?)��ǚ��?19�6Ӓ��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���(�?!$A��Y�?)dY0�G�?1ta�ܖ�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate�qo~�?!h��W���?)|����?1�߲^��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	���"��?!^�f���?)���"��?1^�f���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�F<�͌�?!oT�S�?)���|�r�?1��1�n�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate������?!���#��?)ǟ�lXS�?1w,m$�O�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorۥ���_?!ւ��>�?)ۥ���_?1ւ��>�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[1]::FromTensor]���^?!��-ѵ��?)]���^?1��-ѵ��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��x!^?!gƛK��?)��x!^?1gƛK��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor�/K;5�[?!��z|�?)�/K;5�[?1��z|�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor�ɐc�Y?!�mxB��?)�ɐc�Y?1�mxB��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicekׄ�ƠS?!�hô�ע?)kׄ�ƠS?1�hô�ע?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�`��O?!idY��?)�`��O?1idY��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSlice�#����N?!;�o���?)�#����N?1;�o���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��M�qJ?!�؜��b�?)��M�qJ?1�؜��b�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice��-�H?!9
t�ϗ?)��-�H?19
t�ϗ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 53.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9mD��1�J@I��jz�G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(F�̱�?(F�̱�?!(F�̱�?      ��!       "      ��!       *      ��!       2	�l�?3��?�l�?3��?!�l�?3��?:      ��!       B      ��!       J	Vc	k��?Vc	k��?!Vc	k��?R      ��!       Z	Vc	k��?Vc	k��?!Vc	k��?b      ��!       JCPU_ONLYYmD��1�J@b q��jz�G@