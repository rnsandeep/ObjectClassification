
max_epoch=20
classes=2
dataset='/home/cureskin/HealloAI/preprocessXML/openpores'

exp='pores1'
meanfile='mean_std_darkcircle.npy'

####### Training Network #########################

python train_inceptionv3.py $dataset $exp $meanfile 299 $classes $max_epoch

########## TEsting NEtwork #######################
max_size=320
min_size=240
for epoch in `seq 0 $max_epoch`
do
   for size in `seq $min_size 20 $max_size`	
   do	   
       f="_checkpoint.pth.tar"
       echo "epoch: ", $epoch, "resize_size: ", $size
       python test_inceptionv3.py $dataset $exp/$epoch$f $meanfile 299 $size $classes out_$exp
   done    
done

