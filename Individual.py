import random as random
from copy import deepcopy
import sys as sys

class Individual:
    mutation_steps = [-2, -1, +1, +2]

    def __init__(self, size_fixed_part, min_size_var_part, max_size_var_part, fixed_part_max_values, var_part_max_value):
        self.fixed_part = [0] * size_fixed_part
        self.size_fixed_part = size_fixed_part
        self.var_part = []
        self.fixed_part_max_values = fixed_part_max_values
        self.var_part_max_value = var_part_max_value
        self.min_size_var_part = min_size_var_part
        self.max_size_var_part = max_size_var_part
        self.fitness = None

    @staticmethod
    def random(size_fixed_part, min_size_var_part, max_size_var_part, fixed_part_max_values, var_part_max_value):
        indv = Individual(size_fixed_part, min_size_var_part, max_size_var_part, fixed_part_max_values, var_part_max_value)
        # Step 1. Randomize the fixed part considering the maximum values for each element.
        for i in range(size_fixed_part-1):
            indv.fixed_part[i] = random.randint(0, fixed_part_max_values[i])  # Generate number between 0 and a max (inclusive)
        # Step 2. Determine a random size for the variable part of the individual.
        indv.fixed_part[size_fixed_part - 1] = min_size_var_part + random.randint(0, max_size_var_part-2)
        # Step 3. Generate the variable part randomly.
        for i in range(indv.fixed_part[size_fixed_part - 1]):
            indv.var_part.append(random.randint(0, var_part_max_value-1))
        return indv

    def __str__(self):
        return str(self.fixed_part) +" + "+ str(self.var_part)


    def setVarPartSize(self, newSize):
        diff = newSize - len(self.var_part)
        if diff < 0:
            for i in range(-diff):
                self.var_part.pop()  # this can be improved by removing random positions instead of the last element.
        elif diff > 0:
            for i in range(diff):
                self.var_part.append(random.randint(0, self.var_part_max_value-1)) # this can be improved by adding in random positions.
        self.fixed_part[-1] = newSize


    def randomChange(self, value, min_value, max_value):
        step = 1.0 / len(self.mutation_steps)
        r = random.random()
        i=1
        change = -sys.maxsize-1
        while change == -sys.maxsize-1:
            if r < step * i:
                change = self.mutation_steps[i-1]
            i += 1
        new_value = value + change
        if new_value < min_value:
            new_value = min_value
        elif new_value > max_value:
            new_value = max_value
        return new_value


    def infectPosition(self, pos):
        # Step 1. Determine the maximum allowed for the new random number and the old value.
        if pos < len(self.fixed_part)-1:  # Case 1. Infectation in the fixed part of the individual
            max_value = self.fixed_part_max_values[pos]
            old_value = self.fixed_part[pos]
        else:  # Case 2. Infectation in the var part of the individual.
            max_value = self.var_part_max_value
            old_value = self.var_part[pos-len(self.fixed_part)]
        # Step 2. Generate the new value for the infected position.
        new_value = self.randomChange(value=old_value, min_value=0, max_value=max_value)
        # Step 3. Modify the individual accordingly.
        self.setValue(pos=pos, value=new_value)


    def setValue(self, pos, value):
        if pos < len(self.fixed_part) + len(self.var_part):
            if pos < len(self.fixed_part):
                self.fixed_part[pos] = value
            else:
                self.var_part[pos-len(self.fixed_part)] = value
        else:
            raise Exception("Invalid position of the individual: " + str(pos))



    def infect(self, travel_distance):
        mutated = deepcopy(self)
        # Step 1. Infect (or not) the element of the individual that corresponds to the size of the var-part.
        infect_varsize = (random.randint(0, len(self.fixed_part)-1) == 0)  # the probability is inversely proportional to the fixed part size
        if infect_varsize: # Resize the var-part of the individual.
            # If rnd<0.25 => change=-2; else if rnd<0.5 => change=-1; else if rnd<0.75 => change=+1; else => change=+2.
            mutated.setVarPartSize(self.randomChange(value=len(mutated.var_part), min_value=self.min_size_var_part, max_value=self.max_size_var_part))
            travel_distance = -1  # The change made counts and the travel distance decreases accordingly!
        # Step 2. Determine how many elements of the individual will be infected.
        total_size = len(mutated.fixed_part) + len(mutated.var_part)
        if travel_distance < 0:  # Affected elements are ranging from 0 to the length of fixed+var parts (excluding the element that encodes the size of the var-part).
            nmutated = random.randint(0,total_size-1)
        else:
            nmutated = travel_distance
        # Step 3. Infect 'nmutated' random positions of the individual 'mutated'.
        mutated_positions = []
        i = 0
        while i < nmutated:
            pos = random.randint(0, total_size-1)
            # If the position is not mutated previously and it not corresponds
            # to the element that encodes the size of the var-part.
            if len(mutated_positions) == total_size-1:
                break
            elif pos not in mutated_positions and pos!=len(mutated.fixed_part)-1:
                # Step 3.1. Infect the position 'pos' of the individual.
                mutated.infectPosition(pos)
                mutated_positions.append(pos)
                i+=1
        return mutated

    def __eq__(self, other):
        return self.fixed_part==other.fixed_part and self.var_part==other.var_part





