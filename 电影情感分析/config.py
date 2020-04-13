class config:
    def __init__(self):
        self.id=0
        self.batch_size=128
        self.modelid=3
        self.class_num = 2
        self.model_canshu()
        self.early_stopping=5
        self.lr=0.0007
        self.seed=1666


    def lstm_net(self):
        self.stacked_num = 1
        ###lstm
        self.embedding_size=128
        self.fc1=128
        self.hidden_units=128#4*h


    def stack_lstm_net(self):
        self.fc1 = 128
        ####stack_lstm
        self.embedding_size=128
        self.hidden_units=256#4*h
        self.stacked_num=3

    def model_canshu(self):
        if (self.modelid == 2):
            self.lstm_net()
        elif(self.modelid==3):
            self.stack_lstm_net()

    def write(self,s):
        model=['lstm_net','stacked_lstm_net']
        canshu_list=[self.id,self.batch_size,model[self.modelid-2],self.embedding_size,self.fc1,self.hidden_units,self.stacked_num,self.early_stopping]
        canshu_list=[str(x) for x in canshu_list]
        with open('./log.txt','a',encoding='utf-8') as f:
            f.write('| '+' | '.join(canshu_list)+s+'|\n')