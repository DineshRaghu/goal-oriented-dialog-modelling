
def anonymize_data_task3(inputfile, outputfile):
	
	rest_dict={}
	phone_dict={}
	addr_dict={}
	candidates=set()

	out_file = open(outputfile, 'w')

	with open(inputfile) as in_file:
		for line in in_file:
			line=line.strip()
			if line:
				#print line
				nid, line = line.split(' ', 1)
				if '\t' in line:
					u, r = line.split('\t')
					if 'what do you think of this option:' in r:
						r_split=r.split()
						r_split[-1]=rest_dict[r_split[-1]]
						r=' '.join([str(x) for x in r_split])
					candidates.add(r)
					out_file.write(nid + ' ' + u + '\t' + r +'\n')  # python will convert \n to os.linesep
				else:
					line_split = line.split()
					if line_split[0] not in rest_dict:
						rest_dict[line_split[0]] = "rest_name_" + str(len(rest_dict))
					line_split[0] = rest_dict[line_split[0]]

					if 'R_address' == line_split[1]:
						if line_split[2] not in addr_dict:
							addr_dict[line_split[2]] = "addr_" + str(len(addr_dict))
						line_split[2] = addr_dict[line_split[2]]

					if 'R_phone' == line_split[1]:
						if line_split[2] not in phone_dict:
							phone_dict[line_split[2]] = "phone_" + str(len(phone_dict))
						line_split[2] = phone_dict[line_split[2]]

					line = ' '.join([str(x) for x in line_split])
					out_file.write(nid + ' ' + line + '\n')
			else:
				#print str(len(rest_dict))+','+str(len(phone_dict))+','+str(len(addr_dict))
				out_file.write('\n')
				rest_dict={}
				phone_dict={}
				addr_dict={}
	
	out_file.close()

def anonymize_candidates():
	inputfile = '../data/dialog-bAbI-tasks/dialog-babi-candidates.txt'
	outputfile = '../data/dialog-anonymized/dialog-babi-candidates.txt'
	
	alreadyAdded=set()
	out_file = open(outputfile, 'w')
	with open(inputfile) as in_file:
		for line in in_file:
			line_split = line.split()
			if line_split[-1].startswith('resto_') and line_split[-1].endswith('stars') :
				key=' '.join([str(x) for x in line_split[:-1]])
				if key + '_rest_name' not in alreadyAdded:
					alreadyAdded.add(key + '_rest_name')
					for i in range(0,6):
						out_file.write(key + ' rest_name_' + str(i)+'\n')
			elif line_split[-1].startswith('resto_') and line_split[-1].endswith('_phone') :
				key=' '.join([str(x) for x in line_split[:-1]])
				if key + '_phone' not in alreadyAdded:
					alreadyAdded.add(key + '_phone')
					for i in range(0,6):
						out_file.write(key + ' phone_' + str(i)+'\n')
			elif line_split[-1].startswith('resto_') and line_split[-1].endswith('_address') :
				key=' '.join([str(x) for x in line_split[:-1]])
				if key + '_addr' not in alreadyAdded:
					alreadyAdded.add(key + '_addr')
					for i in range(0,6):
						out_file.write(key + ' addr_' + str(i)+'\n')
			else:
				out_file.write(line.strip()+'\n')
	out_file.close()				

def anonymize_data(base_name):
	files_list = []
	files_list.append("-dev.txt")
	files_list.append("-trn.txt")
	files_list.append("-tst.txt")
	files_list.append("-tst-OOV.txt")

	for file in files_list:
		infile = '../data/dialog-bAbI-tasks/' + base_name + file
		outfile = '../data/dialog-anonymized/' + base_name + file
		anonymize_data_task3(infile,outfile)

if __name__ == '__main__':
	anonymize_data('dialog-babi-task3-options')
	anonymize_candidates()
	