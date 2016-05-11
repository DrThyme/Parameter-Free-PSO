import glob, os, subprocess

for filename in glob.glob("output/output*.txt"):
	with open(filename) as f:
		print "File: "+filename
		avg_sum = 0
		for line in f:
			if "(" in line:
				number = line.replace("(", "").replace(",)", "")
				avg_sum += float(number)
				avg_sum = avg_sum/2

		print "Average: "+str(avg_sum)