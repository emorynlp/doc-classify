for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "running...w2vdim: $i w2vfilt: $j"
		python train_model.py --w2vsource amazon --w2vdim $i --model nonstaticRT --trainable nonstatic --w2vnumfilters $j > /home/tlee54/score/yoonkim-nonstatic/w2vdim$i+w2vfilt$j.txt
	done
done

for i in 50 100 200 400
do
	for j in 16 32 64 128 256
	do
		echo "running...w2vdim: $i w2vfilt: $j"
		python train_model.py --w2vsource amazon --w2vdim $i --model nonstaticRT --trainable static --w2vnumfilters $j > /home/tlee54/score/yoonkim-static/w2vdim$i+w2vfilt$j.txt
	done
done