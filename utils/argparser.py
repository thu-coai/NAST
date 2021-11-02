from .storage import Storage
import argparse

class ArgParser:
	def __init__(self, *args, **kwargs):
		self.parser = argparse.ArgumentParser(*args, **kwargs)
		

	def add_basic(name, storage):
