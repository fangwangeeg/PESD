'''
###Train Loops###
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class Trainer:
    def __init__(self, model, embed_dim, num_classes, temperature, alpha, learning_rate):
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2))
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    @staticmethod
    def grad_reverse(x):
        return GradReverseFunction.apply(x)

    def train(self, target_train_dataloader, source_dataloader, model_name):

        self.model.eval().to(device)
        self.classifier.train().to(device)
        self.discriminator.train().to(device)
        train_loss = 0.0
        total = 0
        correct = 0
        for (source_data, source_label), (target_train_data, target_train_label) in zip (source_dataloader,target_train_dataloader) :
            source_data, source_label = torch.tensor(source_data).float().to(device), torch.tensor(source_label).long().to(device)
            target_train_data, target_train_label = torch.tensor(target_train_data).float().to(device), torch.tensor(target_train_label).long().to(device)

            self.optimizer.zero_grad()

            if model_name == 'Base':
                feature_target = self.model(target_train_data)
                pre_target = self.classifier(feature_target)
                loss = self.criterion(pre_target, target_train_label)

            elif model_name == 'Adversary':
                input_a, input_b = target_train_data, source_data
                input_all = torch.cat((input_a, input_b), dim=0)
                feature_all = self.model(input_all)

                zero_one_list = [0] * input_a.shape[0] + [1] * input_b.shape[0]
                label_dm = torch.tensor(zero_one_list).to(self.device)

                embed = self.grad_reverse(feature_all)
                pre_domain = self.discriminator(embed)
                loss_d = self.criterion(pre_domain, label_dm)

                feature_target = self.model(target_train_data)
                pre_target = self.classifier(feature_target)
                loss_target = self.criterion(pre_target, target_train_label)

                loss = temperature * loss_target + (1 - temperature) * loss_d

            elif model_name == 'Triplet':
                feature_target = self.model(target_train_data)
                feature_source = self.model(source_data)

                pre_target = self.classifier(feature_target)
                loss_target = self.criterion(pre_target, target_train_label)

                if feature_target.shape == feature_source.shape:
                    similarity_scores = F.cosine_similarity(feature_source, feature_target)
                    labels = (source_label == target_train_label).float()
                    margin = 0.2
                    contrast_loss = torch.mean(labels * (1 - similarity_scores) ** 2 + (1 - labels) * torch.clamp(similarity_scores - margin, min=0) ** 2)
                else:
                    contrast_loss = 0
                loss = temperature * loss_target + (1 - temperature) * contrast_loss

            elif model_name == 'Mixup':
                feature_target = self.model(target_train_data)
                feature_source = self.model(source_data)

                pre_target = self.classifier(feature_target)
                loss_target = self.criterion(pre_target, target_train_label)

                if feature_target.shape == feature_source.shape:
                    lam = np.random.beta(alpha, alpha)
                    input_a, input_b = target_train_data, source_data
                    target_a, target_b = target_train_label, source_label
                    mixed_input = lam * input_a + (1 - lam) * input_b
                    mixed_target = lam * target_a + (1 - lam) * target_b
                    mixed_target = mixed_target.long().to(self.device)
                    mixed_feature = self.model(mixed_input)
                    pre_mixed = self.classifier(mixed_feature)
                    loss_mixed = self.criterion(pre_mixed, mixed_target)

                    label_dm_init = torch.ones((mixed_input.shape[0], 1))
                    mixed_label_dm = torch.cat([label_dm_init * lam, label_dm_init * (1 - lam)], 1).to(self.device)
                    mixed_label_dm_digital = torch.max(mixed_label_dm, 1)[1]
                    mixed_embed = self.grad_reverse(mixed_feature)
                    pre_domain_mixed = self.discriminator(mixed_embed)
                    loss_domain = self.criterion(pre_domain_mixed, mixed_label_dm_digital)

                else:
                    loss_domain = 0
                    loss_mixed = 0
                loss = temperature * loss_target + (1 - temperature) * (loss_mixed + loss_domain)

            elif model_name == 'PESD':
                input_a, input_b = target_train_data, source_data
                input_all = torch.cat((input_a, input_b), dim=0)
                feature_all = self.model(input_all)

                zero_one_list = [0] * input_a.shape[0] + [1] * input_b.shape[0]
                label_dm = torch.tensor(zero_one_list).to(self.device)

                embed = self.grad_reverse(feature_all)
                pre_domain = self.discriminator(embed)
                loss_d = self.criterion(pre_domain, label_dm)

                feature_target = self.model(target_train_data)
                feature_source = self.model(source_data)

                pre_target = self.classifier(feature_target)
                loss_target = self.criterion(pre_target, target_train_label)

                if feature_target.shape == feature_source.shape:
                    similarity_scores = F.cosine_similarity(feature_source, feature_target)
                    labels = (source_label == target_train_label).float()
                    margin = 0.2
                    contrast_loss = torch.mean(labels * (1 - similarity_scores) ** 2 + (1 - labels) * torch.clamp(similarity_scores - margin, min=0) ** 2)

                    lam = np.random.beta(alpha, alpha)
                    input_a, input_b = target_train_data, source_data
                    target_a, target_b = target_train_label, source_label
                    mixed_input = lam * input_a + (1 - lam) * input_b
                    mixed_target = lam * target_a + (1 - lam) * target_b
                    mixed_target = mixed_target.long().to(self.device)
                    mixed_feature = self.model(mixed_input)
                    pre_mixed = self.classifier(mixed_feature)
                    loss_mixed = self.criterion(pre_mixed, mixed_target)

                    label_dm_init = torch.ones((mixed_input.shape[0], 1))
                    mixed_label_dm = torch.cat([label_dm_init * lam, label_dm_init * (1 - lam)], 1).to(self.device)
                    mixed_label_dm_digital = torch.max(mixed_label_dm, 1)[1]
                    mixed_embed = self.grad_reverse(mixed_feature)
                    pre_domain_mixed = self.discriminator(mixed_embed)
                    loss_domain = self.criterion(pre_domain_mixed, mixed_label_dm_digital)

                else:
                    contrast_loss = 0
                    loss_domain = 0
                    loss_mixed = 0
                loss = temperature * (loss_target + loss_mixed) + (1 - temperature) * (loss_domain + loss_d) + 0.01 * contrast_loss

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            _, predicted = pre_target.max(1)
            total += target_train_label.size(0)
            correct += predicted.eq(target_train_label).sum().item()

        accuracy = 100.0 * correct / total
        average_loss = train_loss / len(target_train_dataloader)
        return average_loss, accuracy

    def eval(self, test_dataloader):
        self.model.eval()
        self.classifier.eval()
        test_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                labels = labels.long()
                inputs, labels = inputs.to(device), labels.to(device)
                feature_test = self.model(inputs)
                outputs = self.classifier(feature_test)

                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        average_loss = test_loss / len(test_dataloader)

        return average_loss, accuracy
