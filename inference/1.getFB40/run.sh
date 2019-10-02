mkdir fb40
mkdir wavdir
mkdir pcmdir
mkdir -p after_vad

cp `find "/dataset" -name '*.pcm' -path '*dev*' | grep long` pcmdir
ls pcmdir > dev_list_noVAD.txt 


for x in `cat dev_list_noVAD.txt`;do
   echo $x
   ./pcm2wav 1 16000 16 pcmdir/$x wavdir/`echo $x | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`.wav
   ./apply-vad --frame-len=0.025 --frame-shift=0.01 --energy-thresh=1.5e7 --sil-to-speech-trigger=3 --speech-to-sil-trigger=10 wavdir/`echo $x | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`.wav after_vad/`echo $x | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`.wav
done

find "./after_vad" -name "*.wav" > dev_list_VAD.txt
perl get_fb.pl dev_list_VAD.txt fb40 20
rm -r wavdir
rm -r pcmdir
