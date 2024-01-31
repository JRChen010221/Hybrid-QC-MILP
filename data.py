import os
import numpy as np

root_pth = "jobshop"

class JobshopData:
    def __init__(self, job_num: int, machine_num: int) -> None:
        self.job_num = job_num
        self.machine_num = machine_num
        self.instances = []
        self._extract_data()

    def _extract_data(self) -> None:
        """
        Extracts data from a file and populates the 'instances' attribute with the parsed data. 
        """
        data_pth = os.path.join(root_pth, str(self.job_num) + "_" + str(self.machine_num) + ".txt")
        if os.path.exists(data_pth):
            with open(data_pth, 'r') as file:
                file_content = file.read()
                # split the data 
                groups = file_content.split("\n\n")
                instance_num = int((len(groups) - 1) / 4)
                for i in range(instance_num):
                    data = []
                    for group in groups[1+i*4:1+(i+1)*4]:
                        lines = group.strip().split('\n')
                        matrix = [list(map(int, line.split())) for line in lines]
                        data.append(np.array(matrix))
                    self.instances.append(data)
        else:
            raise FileNotFoundError("Data file not found!")