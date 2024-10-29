import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

dataset = 'cora'
id = 'nc_mp'

folder_path = f"output_res/{dataset}/{id}"
predictions = []
labels = []
num = 0

if 'lp' in id:
    label_dict = ['connected', 'unconnected', 'yes', 'no']
elif dataset == 'cora':
    label_dict = ['case_based', 'genetic_algorithms', 'neural_networks', 'probabilistic_methods', 'reinforcement_learning', 'rule_learning', 'theory']
elif dataset == 'pubmed':
    label_dict = ['diabetes mellitus experimental', 'diabetes mellitus type1', 'diabetes mellitus type2']
elif dataset == 'arxiv':
    label_dict = ['cs.NA(Numerical Analysis)', 'cs.MM(Multimedia)', 'cs.LO(Logic in Computer Science)', 'cs.CY(Computers and Society)', 'cs.CR(Cryptography and Security)', 'cs.DC(Distributed, Parallel, and Cluster Computing)', 'cs.HC(Human-Computer Interaction)', 'cs.CE(Computational Engineering, Finance, and Science)', 'cs.NI(Networking and Internet Architecture)', 'cs.CC(Computational Complexity)', 'cs.AI(Artificial Intelligence)', 'cs.MA(Multiagent Systems)', 'cs.GL(General Literature)', 'cs.NE(Neural and Evolutionary Computing)', 'cs.SC(Symbolic Computation)', 'cs.AR(Hardware Architecture)', 'cs.CV(Computer Vision and Pattern Recognition)', 'cs.GR(Graphics)', 'cs.ET(Emerging Technologies)', 'cs.SY(Systems and Control)', 'cs.CG(Computational Geometry)', 'cs.OH(Other Computer Science)', 'cs.PL(Programming Languages)', 'cs.SE(Software Engineering)', 'cs.LG(Machine Learning)', 'cs.SD(Sound)', 'cs.SI(Social and Information Networks)', 'cs.RO(Robotics)', 'cs.IT(Information Theory)', 'cs.PF(Performance)', 'cs.CL(Computational Complexity)', 'cs.IR(Information Retrieval)', 'cs.MS(Mathematical Software)', 'cs.FL(Formal Languages and Automata Theory)', 'cs.DS(Data Structures and Algorithms)', 'cs.OS(Operating Systems)', 'cs.GT(Computer Science and Game Theory)', 'cs.DB(Databases)', 'cs.DL(Digital Libraries)', 'cs.DM(Discrete Mathematics)']
elif dataset == 'instagram':
    label_dict = ['normal','commercial']
for i in range(len(label_dict)):
    label_dict[i] = label_dict[i].lower()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".json"):
        with open(file_path, "r") as file:
            json_data = json.load(file)
            for text in json_data:
                if dataset == 'pubmed':
                    if 'experimental' in text["res"]:
                        text["res"] = 'diabetes mellitus experimental'
                    elif 'type2' in text["res"] or 'type 2' in text["res"]:
                        text["res"] = 'diabetes mellitus type2'
                    elif 'type1' in text["res"] or 'type 1' in text["res"]:
                        text["res"] = 'diabetes mellitus type1'
                res = text["res"].lower().replace("\\", "")
                label = text["gt"].lower()
                if res not in label_dict:
                    print(res, "not in the label dict:", )
                predictions.append(res)
                labels.append(label)
                num+=1

print("len:", len(labels))
print("acc:",accuracy_score(labels, predictions))
print("f1:",f1_score(labels, predictions, average='macro'))

