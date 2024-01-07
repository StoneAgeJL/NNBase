import os
import wandb

class LoggerWriter(object):
    
    def __init__(self, name="experiment", password=None, config=None):
        
        def wrt_netrc(password):
            netrc_path = "/home/hadoop-mtcv/.netrc"
            netrc_cont = "machine api.wandb.ai\n\tlogin user\n\tpassword " + str(password)
            with open(netrc_path, "w") as f:
                f.write(netrc_cont)
                
        if password is not None:
            wrt_netrc(password)
    
        self.wb_logger = wandb.init(
            project=name,
            entity=None,
            # resume="allow",
            resume=None,
            config=config
        )
        
    def step(self, info):
        self.wb_logger.log(info)
        
    def rec(self, name="training_record", ckpt=None, config={}):
        model_artifact = wandb.Artifact(name, type='model', metadata=dict(config))
        if os.path.isfile(ckpt):
            model_artifact.add_file(ckpt)
        self.wb_logger.log_artifact(model_artifact)
        
    def save(self, path):
        if os.path.isfile(path):
            self.wb_logger.save(path)
    
    def quit(self):
        self.wb_logger.finish()
        wandb.finish()
