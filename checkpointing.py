import torch 


def save_checkpoint(model, checkpoint, iteration, out): 

    model_obj = model.state_dict()
    optim_obj = checkpoint.state_dict() 

    data = {
        "model": model_obj,
        "checkpoint": optim_obj, 
        "iteration": iteration
    }

    torch.save(data, out)



def load_checkpoint(src, model, optimizer): 
    
    data = torch.load(src)

    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["checkpoint"])

    return data["iteration"]