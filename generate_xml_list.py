import os

xml_dir = "./data/sp_ppe/xml"
txt_dir = "./data/sp_ppe"

xml_filenames = os.listdir(xml_dir)
txt_file = open(os.path.join(txt_dir, "xml_list.txt"), "w")
for filename in xml_filenames:
    txt_file.write(filename + "\n")
txt_file.close()