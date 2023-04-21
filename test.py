import diffuserSR3.dataset as data 

opt={"name": "Flair",
        "dataroot": "/share/projects/ottopia/superstable/sr3/flair/data/train_64_512",
        "l_resolution": 64,
        "r_resolution": 512,
        "use_shuffle": True,
        "data_len": -1 
    }



dataset = data.create_dataset(opt,'train')

print('dataset created whoo')

opt={"batch_size": 2,
        "use_shuffle": True,
        "num_workers": 1,
    }

dataloadet = data.create_dataloader(dataset,opt,"train")

data1=[]
j=0
for i,batch in enumerate(dataloadet):
    data1.append(next(dataloadet))
    j=+1
    if j>2:
        break

