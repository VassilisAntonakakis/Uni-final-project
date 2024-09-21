import random
import string

class gene:  
    def __init__(self, size, sizeOfGenomes):
        self.size = size
        self.sizeOfGenomes = sizeOfGenomes
        #a gene will be a string of random letters with a length of size * sizeOfGenomes
        self.gene = ''.join(random.choices(string.ascii_letters, k=size * sizeOfGenomes))
        print(f"Gene: {self.gene} with size {size} and sizeOfGenomes {sizeOfGenomes} created!")
    
    def getGene(self):
        return self.gene
        
    def getGenomes(self):
        genomes = []
        for i in range(0, self.size):
            genomes.append(self.gene[i * self.sizeOfGenomes : (i + 1) * self.sizeOfGenomes])
        return genomes
    
    def getSpecificGenome(self, genomeNumber):
        #the value of a gene is the sum of the ascii values of its characters
        #check if the genome number is valid
        if genomeNumber < 0 or genomeNumber >= self.size:
            print(f"Invalid genome number {genomeNumber}!")
            return -1
        #get the requested genome
        genome = self.gene[genomeNumber * self.sizeOfGenomes : (genomeNumber + 1) * self.sizeOfGenomes]
        return genome

    def getGeneValue(self, genomeNumber):
        #the value of a gene is the sum of the ascii values of its characters
        #check if the genome number is valid
        if genomeNumber < 0 or genomeNumber >= self.size:
            print(f"Invalid genome number {genomeNumber}!")
            return -1
        #get the requested genome
        genome = self.getSpecificGenome(genomeNumber)
        return sum([ord(char) for char in genome])
    
    def evolve(self, mutationRate):
        #mutate the gene by changing a random character in it
        mutatedGene = list(self.gene)
        for i in range(0, self.size):
            if random.random() < mutationRate:
                mutatedGene[i] = random.choice(string.ascii_letters)
        self.gene = ''.join(mutatedGene)
        print(f"Gene evolved! New gene: {self.gene}")