import sys
import os

tempList = [files for files in os.listdir("./") if files.endswith(".txt")]
max_test = 0
max_test_name = ''

for eachFile in tempList:
	dmax = tmax = 0
	with open("./" + eachFile, "r") as fin:
		for line in fin:
			l = line.split()
			if not l: continue

			if l[0].startswith('DEV'):
				d = float(l[-1])
			elif l[0].startswith('TEST'):
				t = float(l[-1])

				if d > dmax:
					dmax = d
					tmax = t
				if t > max_test:
					max_test = t
					max_test_name = eachFile
		print '---', eachFile, '---'
		print 'dev', dmax
		print 'tst', tmax
		print ''

print '<---MAX TEST--->'
print max_test_name, ' : ', max_test
