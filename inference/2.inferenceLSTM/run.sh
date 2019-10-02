mkdir /result
find "../1.getFB40/fb40" -name '*.fb'   >dev_list_fb.txt
perl get_list.pl dev_list_fb.txt   lanKey.txt label_dev_list_fb.txt
python inference.py 
