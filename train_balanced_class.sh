
epoch=20
classes=2
dataset='../darkcircle_gpu2/Dataset1'
dataset2='../darkcircle_gpu2/Dataset2/puffyEyes'

exp='puffyBalanced'
meanfile='mean_std_darkcircle.npy'

python train_inceptionv3.py $dataset2 $exp mean_std_darkcircle.npy 299 2 $epoch

#python train_inception_load.py ../darkcircle_gpu2/Dataset1 pretrained_exp6 mean_std_darkcircle.npy 299 2 20 pretrained_exp/49_checkpoint.pth.tar


max=$epoch
max_size=320
for i in `seq 0 $max`
do
   for j in `seq 240 20 $max_size`	
   do	   
       f="_checkpoint.pth.tar"
       echo "epoch: ", $i, "resize_size: ", $j 
       python test_inceptionv3.py $dataset2 $exp/$i$f $meanfile 299 $j $classes out_$exp
   done    
done

