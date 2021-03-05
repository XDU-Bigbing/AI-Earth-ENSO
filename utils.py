import torch
import config


def get_checkpoint_state(model,
                         optimizer,
                         scheduler,
                         load_path=config.MODEL_LOAD_PATH):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, epoch, optimizer, scheduler


def save_checkpoint_state(epoch,
                          model,
                          optimizer,
                          scheduler,
                          save_path=config.MODEL_SAVE_PATH):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, save_path)


def writelog(logstr):
    with open(config.LOG_FILE_PATH, 'a') as f:
        f.write(logstr + '\n')
