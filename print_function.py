import sys
sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
sys.stdout.flush()
