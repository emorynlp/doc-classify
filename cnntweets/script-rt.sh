for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "semeval static w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource twitter --model w2v --datasource semeval --trainable static --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/semeval_static/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "semeval nonstatic w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource twitter --model w2v --datasource semeval --trainable nonstatic --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/semeval_nonstatic/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "semeval nonstatic lex w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource twitter --model w2v_lex --datasource semeval --trainable nonstatic --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/semeval_nonstatic_lex/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "sst static w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource amazon --model w2v --datasource sst --trainable static --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/sst_static/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "sst nonstatic w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource amazon --model w2v --datasource sst --trainable nonstatic --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/sst_nonstatic/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "sst nonstatic lex w2vdim: $i w2vfilt: $j"
		python train_model_1.py --w2vsource amazon --model w2v_lex --datasource sst --trainable nonstatic --w2vdim $i --w2vnumfilters $j > /home/tlee54/score/sst_nonstatic_lex/w2vdim$i+w2vfilt$j.txt
	done
done