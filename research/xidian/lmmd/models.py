"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn
import torch
import lmmd
import mmd

class M2SFE_mmd(nn.Module):
    def __init__(self):
      super(M2SFE_mmd,self).__init__()
      
      self.mmd_loss = mmd.MMDLoss()
      
      self.feature_extractor = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=50, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=50, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.LeakyReLU())

      self.reconstructor = nn.Sequential(
          nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=256, out_channels=50, kernel_size=3, padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=50, out_channels=2, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(2))


      self.cnn_mapping = nn.Sequential(
              nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(512),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(256),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(128),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(50),
              nn.LeakyReLU())
      
      self.rnn_mapping = nn.LSTM(128, 128, num_layers=2, batch_first=True)
      
      self.classifer = nn.Sequential(
          nn.Linear(in_features=6400, out_features=2048),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=2048, out_features=1024),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=1024, out_features=256),
          nn.Dropout(0.6 ),
          nn.LeakyReLU(),
          nn.Linear(in_features=256, out_features=11))
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            
    def forward(self, source, target, s_label):
        
        shallow_feature_src=self.feature_extractor(source)
        cnn_feature_src = self.cnn_mapping(shallow_feature_src)
        rnn_feature_src,_=self.rnn_mapping(cnn_feature_src)
        rnn_feature_src = rnn_feature_src.contiguous().view(rnn_feature_src.size(0),-1)
        
        shallow_feature_tgt=self.feature_extractor(target)
        cnn_feature_tgt = self.cnn_mapping(shallow_feature_tgt)
        rnn_feature_tgt,_=self.rnn_mapping(cnn_feature_tgt)
        rnn_feature_tgt = rnn_feature_tgt.contiguous().view(rnn_feature_tgt.size(0),-1)
        
        s_pred = self.classifer(rnn_feature_src)
        
        t_label = self.classifer(rnn_feature_tgt)
        
        loss_mmd = self.mmd_loss(rnn_feature_src, rnn_feature_tgt)
        return s_pred, loss_mmd

    def predict(self, x):
        shallow_feature=self.feature_extractor(x)
        cnn_feature = self.cnn_mapping(shallow_feature)
        rnn_feature,_=self.rnn_mapping(cnn_feature)
        rnn_feature = rnn_feature.contiguous().view(rnn_feature.size(0),-1)
        logits = self.classifer(rnn_feature)
        return logits
  

class M2SFE_lmmd(nn.Module):
    def __init__(self):
      super(M2SFE_lmmd,self).__init__()
      
      self.lmmd_loss = lmmd.LMMD_loss(class_num=11)
      
      self.feature_extractor = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=50, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=50, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.LeakyReLU())

      self.reconstructor = nn.Sequential(
          nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=256, out_channels=50, kernel_size=3, padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=50, out_channels=2, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(2))


      self.cnn_mapping = nn.Sequential(
              nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(512),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(256),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(128),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(50),
              nn.LeakyReLU())
      
      self.rnn_mapping = nn.LSTM(128, 128, num_layers=2, batch_first=True)
      
      self.classifer = nn.Sequential(
          nn.Linear(in_features=6400, out_features=2048),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=2048, out_features=1024),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=1024, out_features=256),
          nn.Dropout(0.6 ),
          nn.LeakyReLU(),
          nn.Linear(in_features=256, out_features=11))
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            
    def forward(self, source, target, s_label):
        
        shallow_feature_src=self.feature_extractor(source)
        cnn_feature_src = self.cnn_mapping(shallow_feature_src)
        rnn_feature_src,_=self.rnn_mapping(cnn_feature_src)
        rnn_feature_src = rnn_feature_src.contiguous().view(rnn_feature_src.size(0),-1)
        
        shallow_feature_tgt=self.feature_extractor(target)
        cnn_feature_tgt = self.cnn_mapping(shallow_feature_tgt)
        rnn_feature_tgt,_=self.rnn_mapping(cnn_feature_tgt)
        rnn_feature_tgt = rnn_feature_tgt.contiguous().view(rnn_feature_tgt.size(0),-1)
        
        s_pred = self.classifer(rnn_feature_src)
        
        t_label = self.classifer(rnn_feature_tgt)
        
        loss_lmmd = self.lmmd_loss.get_loss(rnn_feature_src, rnn_feature_tgt, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        shallow_feature=self.feature_extractor(x)
        cnn_feature = self.cnn_mapping(shallow_feature)
        rnn_feature,_=self.rnn_mapping(cnn_feature)
        rnn_feature = rnn_feature.contiguous().view(rnn_feature.size(0),-1)
        logits = self.classifer(rnn_feature)
        return logits
    

class M2SFE_lmmd_wo_bn(nn.Module):
    def __init__(self):
      super(M2SFE_lmmd_wo_bn,self).__init__()
      
      self.lmmd_loss = lmmd.LMMD_loss(class_num=11)
      
      self.feature_extractor = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=50, kernel_size=3,padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=50, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,padding=1,stride=1),
          nn.LeakyReLU())

      self.reconstructor = nn.Sequential(
          nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=256, out_channels=50, kernel_size=3, padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=50, out_channels=2, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(2))


      self.cnn_mapping = nn.Sequential(
              nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,stride=1),

              nn.LeakyReLU(),
              nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,stride=1),

              nn.LeakyReLU(),
              nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,padding=1,stride=1),

              nn.LeakyReLU(),
              nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3,padding=1,stride=1),

              nn.LeakyReLU())
      
      self.rnn_mapping = nn.LSTM(128, 128, num_layers=2, batch_first=True)
      
      self.classifer = nn.Sequential(
          nn.Linear(in_features=6400, out_features=2048),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=2048, out_features=1024),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=1024, out_features=256),
          nn.Dropout(0.6 ),
          nn.LeakyReLU(),
          nn.Linear(in_features=256, out_features=11))
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            
    def forward(self, source, target, s_label):
        
        shallow_feature_src=self.feature_extractor(source)
        cnn_feature_src = self.cnn_mapping(shallow_feature_src)
        rnn_feature_src,_=self.rnn_mapping(cnn_feature_src)
        rnn_feature_src = rnn_feature_src.contiguous().view(rnn_feature_src.size(0),-1)
        
        shallow_feature_tgt=self.feature_extractor(target)
        cnn_feature_tgt = self.cnn_mapping(shallow_feature_tgt)
        rnn_feature_tgt,_=self.rnn_mapping(cnn_feature_tgt)
        rnn_feature_tgt = rnn_feature_tgt.contiguous().view(rnn_feature_tgt.size(0),-1)
        
        s_pred = self.classifer(rnn_feature_src)
        
        t_label = self.classifer(rnn_feature_tgt)
        
        loss_lmmd = self.lmmd_loss.get_loss(rnn_feature_src, rnn_feature_tgt, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        shallow_feature=self.feature_extractor(x)
        cnn_feature = self.cnn_mapping(shallow_feature)
        rnn_feature,_=self.rnn_mapping(cnn_feature)
        rnn_feature = rnn_feature.contiguous().view(rnn_feature.size(0),-1)
        logits = self.classifer(rnn_feature)
        return logits

class model(nn.Module):
    def __init__(self):
      super(model,self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3,padding=1, stride=1),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1,stride=1),
          nn.LeakyReLU(),
          nn.Flatten()
          )
      self.fc = nn.Sequential(
          nn.Linear(in_features=32*128, out_features=128),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=64),
          nn.ReLU(),         
          nn.Linear(in_features=64, out_features=11)
          )
      self.lmmd_loss = lmmd.LMMD_loss(class_num=11)
          
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)    
            
    def forward(self, source, target, s_label):
        source = self.conv(source)
        s_pred = self.fc(source)
        target = self.conv(target)
        t_label = self.fc(target)
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.conv(x)
        return self.fc(x)


class Classifier(nn.Module):
    """ classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Classifier, self).__init__()


    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.fc(feat)
        return out

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
class DNET_Encoder(nn.Module):
    def __init__(self):
      super(DNET_Encoder,self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1, stride=1),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.ReLU())

      self.Rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True)
      
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x):
      x1 = self.conv(x)
      x2,_ = self.Rnn(x1)
      feat = x2.contiguous().view(x2.size(0),-1)
      return feat
  
    def feature(self, x):
      x1 = self.conv(x)
      #x2,_ = self.Rnn(x1)
      feat = x1.contiguous().view(x1.size(0),-1)
      return feat
  
class DNET_Classifier(nn.Module):
    """ classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(DNET_Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=11))

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.fc(feat)
        return out  
    
class DNET_Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(DNET_Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(6400, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
           # nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
    
class DNET(nn.Module):
    def __init__(self):
      super(DNET,self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1, stride=1),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.ReLU())

      self.Rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True)
     
      self.fc = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=11))
      
      self.lmmd_loss = lmmd.LMMD_loss(class_num=11)
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, source, target, s_label):
        source_feat_conv = self.conv(source)
        source_feat,_ = self.Rnn(source_feat_conv)
        s_pred = self.fc(source_feat.contiguous().view(source_feat.size(0),-1))
        
        target_feat_conv = self.conv(target)
        target_feat,_ = self.Rnn(target_feat_conv)
        t_label = self.fc(target_feat.contiguous().view(target_feat.size(0),-1))
        
        loss_lmmd = self.lmmd_loss.get_loss(source_feat.contiguous().view(source_feat.size(0),-1), target_feat.contiguous().view(target_feat.size(0),-1), s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        x1 = self.conv(x)
        #x1 = x1.view(x1.size(0),-1)
        #x2,_ = self.Rnn(x1)
        y = x1.contiguous().view(x1.size(0),-1)
        x = self.fc(y)
        return x
    
    '''def feature(self, x):
        x1 = self.conv(x)
        x2,_ = self.Rnn(x1)
        y = x1.contiguous().view(x1.size(0),-1)
        return y  '''  
    
    
    