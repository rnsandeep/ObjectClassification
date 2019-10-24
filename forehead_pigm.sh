
max_epoch=120
classes=2
dataset='/home/cureskin/CLASSIFIERS_DATA/forehead_data/forehead_cc'

exp='foreheadpigm_cc'
meanfile='mean_std_darkcircle.npy'

####### Training Network #########################

python3 train_inceptionv3_clr_weighed.py $dataset $exp $meanfile 299 $classes $max_epoch

########## TEsting NEtwork #######################
max_size=300
min_size=300
for epoch in `seq 0 $max_epoch`
do
   for size in `seq $min_size 20 $max_size`	
   do	   
       f="_checkpoint.pth.tar"
       echo "epoch: ", $epoch, "resize_size: ", $size
       python3 test_inceptionv3.py $dataset $exp/$epoch$f $meanfile 299 $size $classes out_$exp
   done    
done

