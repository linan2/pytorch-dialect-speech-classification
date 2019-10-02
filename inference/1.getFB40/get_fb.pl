use strict;

(my $FileIn, my $Des, my $blocks) = @ARGV;

my $dir_bin = "./bin";

open(IN, "$FileIn");
my @lists = ();
while(my $line = <IN>)
{
	chomp($line);
	my @array = split/\/|\./, $line;
	my $cur_list = "$line $Des/$array[$#array-1]\.fb";
	push(@lists, $cur_list);
}
close(IN);

mkdir("LST");
my $total_num = scalar(@lists);
my $block_num = int($total_num/$blocks)+1;
for(my $i = 0; $i < $blocks; $i++)
{
	open(OUT, ">LST/fea_list$i.scp");
	for(my $j = 0; $j < $block_num; $j++)
	{
		my $sub_index = $i*$block_num+$j;
		if($sub_index == @lists)
		{
			last;
		}
		print OUT "$lists[$sub_index]\n";
	}
	close(OUT);
	
	my $cmd = "$dir_bin/HCopy -C $dir_bin/HCopy_config -S LST/fea_list$i.scp";
	print "$cmd\n";
	system("$cmd");
}

print "Done!\n";
