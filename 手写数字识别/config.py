from datetime import datetime
class config:
    def __init__(self):

        self.conv_struct=[1,16,32]
        self.kernel_size = [0,3,3]
        self.conv_dropout = [0, 0, 0]
        self.pool=[0,2,2]
        self.stride = [0, 1, 1]
        # self.padding = [0, 1, 1]

        self.val_id=5

        self.gamma=0.9
        self.EPOCH=1
        self.LR=0.01
        self.BATCH_SIZE=64
        self.early_stopping=5
        self.time=str(datetime.now())




    def string(self):
        s=self.get_title()+"| "+self.time+" | "+str(0)+" | "+str(0)+" | "+str(0)+" | " +str(self.conv_struct)+" | "+str(self.kernel_size)+" | "+str([self.pool,self.stride])+" | "+str(self.conv_dropout)+" | "+str(self.EPOCH)+" | "+str(self.LR)+" | "+str(self.BATCH_SIZE)+" | "+str(self.early_stopping)+" |"
        return s
    def write_string(self,filepath,train_acc,max_acc,F1):
        s = "| " + self.time + " | " + str(train_acc)+" | "+str(max_acc)+" | "+str(F1) + " | " + str(
            self.conv_struct) + " | " + str(self.kernel_size) + " | " + str(
            [self.pool, self.stride]) + " | " + str(self.conv_dropout) + " | " + str(
            self.EPOCH) + " | " + str(self.LR) + " | " + str(self.BATCH_SIZE) + " | " + str(self.early_stopping) + " |\n"
        with open(filepath,'a',encoding='utf-8') as f:
            f.write(s)

    def get_title(self):
        s='| time | train_acc | max_acc | mean_acc | conv_struct | kernel_size | [pool-stride] | conv_dropout | gamma | EPOCH | LR | BATCH_SIZE | early_stopping |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|\n'
        return s

    def print_experiment(self,filepath,experiment):
        s=experiment+self.get_title()
        with open(filepath,'a',encoding='utf-8') as f:
            f.write(s)

# opt=config()
# opt.string()