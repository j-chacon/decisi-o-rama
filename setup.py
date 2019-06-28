from distutils.core import setup

setup(
      name='decisiorama',
      version='0.1',
      description='A libraryfor preference elicitacion in MCDA',
      author='Juan Carlos Chacon-Hurtado',
      author_email='j.chaconhurtado@tudelft.nl',
      url='None',
      packages=['decisiorama', 
                'decisiorama.pda',
                'decisiorama.sensitivity',
                'decisiorama.utils'],
     )