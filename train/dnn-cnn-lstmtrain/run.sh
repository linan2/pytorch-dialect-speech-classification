# module load cuda/9.1.1
# export PATH="/home/sre/jinli/anaconda2/bin:$PATH"
#if [ ! -d ./log ];then
#mkdir log 
#fi

# 获得做完VAD后训练集和开发集的语音数据列表文件
find "/home/train02/linan/ASR/keda/newtrain/aichallenge/train/1.getFB40/fb40" -name '*train*.fb' >train_list_fb.txt
find "/home/train02/linan/ASR/keda/newtrain/aichallenge/train/1.getFB40/fb40" -name '*dev*.fb'   >dev_list_fb.txt


# 转换成带标签的列表
perl get_list.pl train_list_fb.txt lanKey.txt label_train_list_fb.txt
perl get_list.pl dev_list_fb.txt   lanKey.txt label_dev_list_fb.txt

# 训练网络，参数在train.py里面配置
#nohup python train.py  >log/TrainLan.log 2>&1&

python train.py 
