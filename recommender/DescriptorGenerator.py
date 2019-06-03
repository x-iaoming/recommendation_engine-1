import os
os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'
from chemdescriptor import ChemAxonDescriptorGenerator


class DescriptorGenerator

	# all combos is an array of all triples, amounts, and grid parameteres 
	#that can be obtained from the Generator class

	def __init__(self,smilefile,desfile):
		self.smilefile = smilefile
		self.desfile = desfile

	def generateDescriptor(self):

		cag = ChemAxonDescriptorGenerator(self.smilefile,
                                  self.desfile,
                                  ph_values=[7],
                                  command_stems=None,
                                  ph_command_stems=None)

		cag.generate('opnew.csv')


smilef = '/home/h205c/chemdescriptor/examples/test_foursmiles.smi'
desf = '/home/h205c/chemdescriptor/examples/descriptors_list.json'
dg = DescriptorGenerator(smilef, desf)
dg.generateDescriptor()
