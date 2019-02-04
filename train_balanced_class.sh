
epoch=20
classes=6
dataset='/home/user/Data/Backup/IMAGES-DATA/HIGHER/new'

exp='exp1'
meanfile='mean_std_darkcircle.npy'

python train_inceptionv3.py $dataset $exp mean_std_darkcircle.npy 299 $classes $epoch

max=$epoch
max_size=320
for i in `seq 0 $max`
do
   for j in `seq 240 20 $max_size`	
   do	   
       f="_checkpoint.pth.tar"
       echo "epoch: ", $i, "resize_size: ", $j 
       python test_inceptionv3.py $dataset $exp/$i$f $meanfile 299 $j $classes out_$exp
   done    
done

