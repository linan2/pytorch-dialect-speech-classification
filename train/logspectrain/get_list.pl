use strict;

(my $FileIn, my $FileLAN, my $FileOut) = @ARGV;

my %hash = ();
open(IN, "$FileLAN");
while(my $line = <IN>)
{
	chomp($line);
	my @array = split/\s+/, $line;
	my $lan = $array[0];
	my $target = $array[1];
	if(!exists $hash{$lan})
	{
		$hash{$lan} = $target;
	}
	else
	{
		print "repe : $lan !!\n";
	}
}
close(IN);

open(IN, "$FileIn");
open(OUT, ">$FileOut");
while(my $line = <IN>)
{
	chomp($line);
	my @array = split/\//, $line;
	my @tmp = split/\_/, $array[$#array];
	my $lan = $tmp[0];
	if(exists $hash{$lan})
	{
		print OUT "$line $hash{$lan}\n"
	}
	else
	{
		print "wrong Label : $line !!\n";
	}
}
close(IN);
close(OUT);

print "Done!\n";
