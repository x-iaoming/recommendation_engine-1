class Model_Interpreter: 
	#Sieve extracts successful reactions predicted by machine learning model
	def _init_(self,desired_desc,model)
	    """
	 	Args: A dataframe of possible reactions generated by Generate class
	    """
		self.model = model
		self.desired_desc = desired_desc

	def filterDesc(self,dataframe)
	 	"""
	 	Args: A dataframe of possible reactions generated by Generate class
	    """
	    

	    for descriptor in self.desired_desc

	    reactions['C1descriptor']
	    reactions['C2descriptor']
	    reactions['C3descriptor']
		for reaction in reactions:
			if reaction