import glob
import pdb


class timeReset:
	def __init__(self):
		suf = ["p3", "p7", "p11", "p15"]
		delay_suf = ["p2", "p6", "p10", "p14"]
		self.path = 'D:\\imMens Learning\\Faa_neew\\'
		self.raw_files = glob.glob('*imMensEvt.txt')
		self.cur_files = []
		self.timeroot = dict()
		self.timeroot['p3-1-0-imMensEvt.txt'] = '1377029476722'
		self.timeroot['p7-1-0-imMensEvt.txt'] = '1378230408618'
		self.timeroot['p11-1-0-imMensEvt.txt'] = '1378424050515'
		self.timeroot['p15-1-0-imMensEvt.txt'] = '1379967316627'

		for users in suf:
			for files in self.raw_files:
				if files.find(users) != -1:
					print(files)
					self.cur_files.append(files)

	#Helper function to get the format hh:mm:ss by padding things that are less than 0
	def zeropad(self, number):
		if number < 10:
			number = "0"+str(number)
		else:
			number = str(number)
		return number

	#takes miliseconds and converts it into hh:mm:ss
	def convertMili(self, miliseconds):
		seconds = miliseconds/1000
		if(seconds<1):
			return "00:00:00"
		else:
			hours = int(seconds/3600)
			remain = seconds % 3600
			hours = self.zeropad(hours)
			minutes = int(remain/60)
			minutes = self.zeropad(minutes)
			seconds = remain % 60
			seconds = self.zeropad(int(seconds))
			time = hours + ":" + minutes + ":"+ seconds
			return time


	def offsetTime(self, base, time):
		time2 = time-base
		# pdb.set_trace()
		return self.convertMili(time2)

	#returns a list of converted times in hh:mm:ss format
	# def grabLines(filename):
	# 	file = open(filename, 'r')
	# 	lines = file.readlines()
	# 	base = int(lines[0].split(",")[1])
	# 	count =0
	# 	for i in range(len(lines)):
	# 		cols = lines[i].split(",")
	# 		cols[1] = offsetTime(base, int(cols[1]))
	# 		lines[i] = ",".join(cols)
	# 	file.close()
	# 	return lines

	def grabLines(self, filename, baseline):
		file = open(filename, 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			cols = lines[i].split(",")
			cols[1] = self.offsetTime(int(baseline), int(cols[1]))
			lines[i] = ",".join(cols)
		file.close()
		return lines

	#creates a new file with edited time with name edited attached
	# def writeFile(filename):
	# 	editedLines = grabLines(filename)

	# 	#pick whatever name you want for these edited files
	# 	elems = filename.split("/")
	# 	elems[2] = "edited-" + elems[2]
	# 	filename = "editedTime/" + elems[2]
	# 	file = open(filename, "w")
	# 	for line in editedLines:
	# 		file.write(line)
	# 	file.close()
	def writeFile(self, filename, baseline, outputPath):
		editedLines = self.grabLines(filename, baseline)
		file = open(outputPath, "w")
		for line in editedLines:
			file.write(line)
		file.close()

if __name__ == "__main__":
	obj = timeReset()
	for files in obj.cur_files:
		print(files)
		fname = files.split(".")[0]
		fname += '_fix.txt'
		print(fname)
		obj.writeFile(files, obj.timeroot[files], fname)
		# break
