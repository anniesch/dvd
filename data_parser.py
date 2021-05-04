import os
import json
import numpy as np

from collections import namedtuple
from collections import defaultdict

ListData = namedtuple('ListData', ['id', 'label', 'path'])


class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, args, json_path_input, json_path_labels, data_root,
                 extension, num_tasks, is_test=False, is_val=False):
        self.num_tasks = num_tasks
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test
        self.is_val = is_val
        self.just_robot = args.just_robot
        self.sim_dir = args.sim_dir
        
        self.num_occur = defaultdict(int)
        
        self.tasks = args.human_tasks
        self.add_demos = args.add_demos
        if self.add_demos:
            self.robot_tasks = args.robot_tasks

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()
        print("Number of human videos:", self.num_occur.values())
        
        
    def read_json_input(self):
        json_data = []
        if not self.is_test:
            if not self.just_robot: #not self.triplet or not self.add_demos: #self.is_val or
                with open(self.json_path_input, 'rb') as jsonfile:
                    json_reader = json.load(jsonfile)
                    for elem in json_reader:
                        label = self.clean_template(elem['template'])
                        if label not in self.classes_dict.keys(): # or label == 'Pushing something so that it slightly moves':
                            continue
                        if label not in self.classes:
                            raise ValueError("Label mismatch! Please correct")
                        
                        label_num = self.classes_dict[label]
                        item = ListData(elem['id'],
                                        label,
                                        os.path.join(self.data_root,
                                                     elem['id'] + self.extension)
                                        )
                        json_data.append(item)
                        self.num_occur[label] += 1
            
            if self.add_demos: 
                # Add robot demonstrations or extra robot class to json_data, just use id 300000
                robot_tasks = self.robot_tasks
                root_in_dir = self.sim_dir 
                for label_num in robot_tasks: 
                    # add task demos for task label_num
                    in_dirs = [f'{root_in_dir}/env1/task{label_num}_webm', f'{root_in_dir}/env1_rearranged/task{label_num}_webm']
                        
                    for in_dir in in_dirs:
                        label = self.classes_dict[label_num]

                        num_demos = self.add_demos
                        self.num_occur[label] += num_demos
                        if not self.is_val: 
                            for j in range(num_demos):
                                item = ListData(300000,
                                            label,
                                            os.path.join(in_dir, str(j) + self.extension)
                                            )
                                json_data.append(item)
                        else:
                            for j in range(num_demos, int(1.4*num_demos)):
                                item = ListData(300000,
                                            label,
                                            os.path.join(in_dir, str(j) + self.extension)
                                            )
                                json_data.append(item)
                        

        else:
            with open(self.json_path_input, 'rb') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {} 
        tasks = self.tasks
        for i, item in enumerate(classes):
            if i not in tasks:
                continue
            classes_dict[item] = i
            classes_dict[i] = item
        print("Length of keys", len(classes_dict.keys()), classes_dict.keys())
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class WebmDataset(DatasetBase):
    def __init__(self, args, json_path_input, json_path_labels, data_root, num_tasks, 
                 is_test=False, is_val=False):
        EXTENSION = ".webm"
        super().__init__(args, json_path_input, json_path_labels, data_root,
                         EXTENSION, num_tasks, is_test, is_val)


class I3DFeatures(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".npy"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)


class ImageNetFeatures(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".npy"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)
