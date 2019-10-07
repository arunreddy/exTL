from intl.data.amazon_dataset import AmazonDataset
from torch.utils.data import DataLoader

def main():



  source_domain = 'dvd'
  target_domain = 'electronics'

  nDim = 4000

  tr_data = AmazonDataset(source=source_domain, target=target_domain, partition_name='train', target_train_ratio=0.05, max_features=nDim)
  te_data = AmazonDataset(source=source_domain, target=target_domain, partition_name='test', target_train_ratio=0.05, max_features=nDim)
  va_data = AmazonDataset(source=source_domain, target=target_domain, partition_name='validation', target_train_ratio=0.05, max_features=nDim)


  tr_dataloader = DataLoader(tr_data, shuffle=True, batch_size=50)
  te_dataloader = DataLoader(te_data, batch_size=50)
  va_dataloader = DataLoader(va_data, shuffle=True, batch_size=50)





if __name__ == '__main__':
  main()
