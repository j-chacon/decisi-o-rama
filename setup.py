from distutils.core import setup

setup(
      name='decisiorama',
      version='0.1',
      description='Decisi-o-rama: An open-source Python library for multi-attribute value/utility decision analysis',
      author='Juan Carlos Chacon-Hurtado',
      author_email='j.chaconhurtado@tudelft.nl',
      url='None',
      packages=['decisiorama', 
                'decisiorama.pda',
                'decisiorama.sensitivity',
                'decisiorama.utils'],
     )
