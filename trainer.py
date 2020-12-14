import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
import torchvision
from wide_resnet import Wide_ResNet
from tqdm import tqdm
from visualize_features import plot_pca, plot_tsne
import argparse
import os
import functools
from copy import deepcopy
import itertools

def compute_accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    correct = (pred == y)
    acc = correct.sum().float() / len(y)
    return float(acc), correct


class Trainer(object):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(Trainer, self).__init__()
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = SummaryWriter(log_dir=args.logdir, flush_secs=60)
        self.early_stop = False
        self.tracked_metric = 'train_loss'
        self.metric_comparator = lambda x, y: x<y

    def track_metric(self, metric_name, direction):
        self.tracked_metric = metric_name
        if direction == 'min':
            self.best_metric = float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x<y
        else:
            self.best_metric = -float('inf')
            self.epochs_since_best = 0
            self.metric_comparator = lambda x,y: x>y

    def _log(self, logs, step):
        for k,v in logs.items():
            self.logger.add_scalar(k, v, global_step=step)

    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            output, logs = func(*args, **kwargs)
            output['loss'].backward()
            self.optimizer.step()
            return output, logs
        return wrapper

    def criterion(self, logits, y):
        if self.args.loss_type == 'xent':
            loss = 1.0 * nn.functional.cross_entropy(logits, y, reduction='none')
        else:
            raise NotImplementedError(self.args.loss_type)
        return loss    

    def train_step(self, batch, batch_idx):
        x,y = batch        
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc, correct = compute_accuracy(logits.detach().cpu(), y.detach().cpu())
        
        loss = loss.mean()

        return {'loss':loss}, {'train_accuracy': acc,
                             'train_loss': float(loss.detach().cpu())}
    
    def val_step(self, batch, batch_idx):        
        output, logs = self.train_step(batch, batch_idx)
        output['loss'] = output['loss'].detach().cpu()
        val_logs = {'lr':self.scheduler.optimizer.param_groups[0]['lr']}
        for k,v in logs.items():
            val_logs[k.replace('train', 'val')] = v
        return output, val_logs
    
    def test_step(self, batch, batch_idx):
        output, logs = self.train_step(batch, batch_idx)
        output['loss'] = output['loss'].detach().cpu()
        test_logs = {}
        for k,v in logs.items():            
            test_logs[k.replace('train', 'test')] = v
        return output, test_logs

    def _batch_loop(self, func, loader, epoch_idx):
        t = tqdm(enumerate(loader))
        t.set_description('epoch %d' % epoch_idx)
        all_outputs = []
        for i, batch in t:            
            outputs, logs = func(batch, i)
            all_outputs.append(outputs)
            self._log(logs, i + epoch_idx*len(loader))
            if 'metrics' not in locals():
                metrics = {k:0 for k in logs.keys()}
            for k,v in logs.items():
                metrics[k] = (i*metrics[k] + v)/(i+1)
            t.set_postfix(**metrics, best_metric=self.best_metric)
            if self.args.debug and (i == 5):
                break
        return all_outputs, metrics

    def train_loop(self, epoch_idx, post_loop_fn=None):
        self.model = self.model.train()
        outputs, metrics = self._batch_loop(self._optimization_wrapper(self.train_step),
                                    self.train_loader, epoch_idx)
        if post_loop_fn is not None:                                    
            outputs = post_loop_fn(outputs, metrics, epoch_idx)
        return outputs, metrics

    def val_loop(self, epoch_idx, post_loop_fn=None):
        self.model = self.model.eval()
        outputs, metrics = self._batch_loop(self.val_step, self.val_loader, epoch_idx)
        if post_loop_fn is not None:                                    
            outputs = post_loop_fn(outputs, metrics, epoch_idx)
        return outputs, metrics

    def test_loop(self, post_loop_fn=None):
        self.model = self.model.eval()
        outputs, metrics = self._batch_loop(self.test_step, self.test_loader, 0)
        if post_loop_fn is not None:                                    
            outputs, metrics = post_loop_fn(outputs, metrics)
        return outputs, metrics

    def create_or_clear_cpdir(self, metric, epoch_idx):
        outdir = os.path.join(self.args.logdir, 'checkpoints')
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            for fn in os.listdir(outdir):
                os.remove(os.path.join(outdir, fn))

        outfile = os.path.join(outdir,
                                "metric=%.2f-epoch=%d.pth" % (metric, epoch_idx))
        print("The outfile directory is:", outfile)
        return outfile

    def checkpoint(self, metric, epoch_idx, comparator):
        if comparator(metric, self.best_metric):
            self.best_metric = metric
            self.epochs_since_best = 0

            outfile = self.create_or_clear_cpdir(metric, epoch_idx)
            torch.save(self.model, outfile)
            self.best_checkpoint = outfile
        else:
            self.epochs_since_best += 1

    def check_early_stop(self):
        if self.epochs_since_best > 3*self.args.patience:
            self.early_stop = True
    
    def epoch_end(self, epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics):
        metrics = train_metrics
        metrics.update(val_metrics)

        self.checkpoint(metrics[self.tracked_metric], epoch_idx, self.metric_comparator)
        self.check_early_stop()
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics[self.tracked_metric])
        else:
            self.scheduler.step()
    
    def train_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics
    
    def val_epoch_end(self, outputs, metrics, epoch_idx):
        return outputs, metrics

    def test_epoch_end(self, outputs, metrics):
        return outputs, metrics

    def train(self):
        torch.save(self.args, os.path.join(self.args.logdir, 'args.pkl'))

        for i in range(self.args.nepochs):
            train_output, train_metrics = self.train_loop(i, post_loop_fn=self.train_epoch_end)
            val_output, val_metrics = self.val_loop(i, post_loop_fn=self.val_epoch_end)
            self.epoch_end(i, train_output, val_output, train_metrics, val_metrics)
            
            if self.early_stop:
                break

        metrics = train_metrics
        metrics.update(val_metrics, post_loop_fn=self.test_epoch_end)
        self.logger.add_hparams(dict(vars(self.args)), {self.tracked_metric: metrics[self.tracked_metric]})
        self.model = torch.load(self.best_checkpoint)

    def test(self):        
        _, test_metrics = self.test_loop(post_loop_fn=self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)

class AETrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(AETrainer, self).__init__(model, train_loader, val_loader, 
                                                test_loader, optimizer, scheduler, 
                                                device, args)
    
    def get_averaged_images(self, x):
        b, c, h, w = x.shape
        wts = torch.rand((self.args.num_averaged_training_images, b), device=x.device)
        wts /= torch.sum(wts, dim=1, keepdim=True)
        avg_pairs = torch.mm(wts, x.view(b, -1))
        avg_pairs = avg_pairs.view(self.args.num_averaged_training_images, c, h, w)
        return avg_pairs

    def criterion(self, x, x_):
        if not hasattr(self.args, 'loss_type') or self.args.loss_type == 'mse':
            loss =  nn.functional.mse_loss(x, x_, reduction='none').view(x.shape[0],-1).mean(1)            
        elif self.args.loss_type == 'bce':
            loss = nn.functional.binary_cross_entropy(x_, x, reduction='none')
            loss = loss.view(loss.shape[0], -1).mean(1)            
        else:
            raise ValueError(self.args.loss_type)
        return loss

    def train_step(self, batch, batch_idx):
        x,_ = batch
        x = x.to(self.device)

        if self.args.num_averaged_training_images > 0:
            avg_pairs = self.get_averaged_images(x)
            x = torch.cat((x, avg_pairs), dim=0)
        fx, x_ = self.model(x)

        loss = self.criterion(x, x_)

        if self.args.use_feature_recon_loss:
            fx_ = self.ae.encode(x_)
            feature_loss = self.criterion(fx.detach(), fx_)
            loss += self.args.feature_recon_weight * feature_loss
        loss = loss.mean()
        
        logs = {'train_loss': float(loss.detach().cpu())}
        return {'loss': loss}, logs

    def val_step(self, batch, batch_idx):
        x,_ = batch
        x = x.to(self.device)

        fx, x_ = self.model(x)        
        loss = self.criterion(x, x_).mean()
        if self.args.use_feature_recon_loss:
            fx_ = self.ae.encoder(x_)
            loss += self.criterion(fx, fx_).mean()
        loss = loss.detach().cpu()
        return {'val_loss': loss}, {'val_loss': float(loss.mean())}
    
    def test_step(self, batch, batch_idx):
        x,_ = batch
        x = x.to(self.device)

        fx = self.model.encode(x)
        x_ = self.model.decode(fx)

        loss_per_image = self.criterion(x, x_)
        if self.args.use_feature_recon_loss:
            fx_ = self.ae.encoder(x_)
            loss_per_image += self.criterion(fx, fx_)

        loss = loss_per_image.detach().cpu()
        logs = {'test_loss': float(loss.mean())}
        
        return {
            'test_loss': loss,
            'images':{
                'x': x.detach().cpu(),
                'x_': x_.detach().cpu()
            }
        }, logs
    
    def test_epoch_end(self, outputs, metrics):    
        loss = torch.cat([x['test_loss'] for x in outputs], dim=0)        
        originals = torch.cat([x['images']['x'] for x in outputs], dim=0)
        recon = torch.cat([x['images']['x_'] for x in outputs], dim=0)

        avg_loss = loss.mean()

        n_images = 5
        _, best_idxs = torch.topk(loss, n_images, largest=False)
        _, worst_idxs = torch.topk(loss, n_images, largest=True)
        random_idxs = torch.randint(len(originals), (n_images,))

        random_imgs = torch.cat((originals[random_idxs], recon[random_idxs]), dim=0)
        best_imgs = torch.cat((originals[best_idxs], recon[best_idxs]), dim=0)
        # best_imgs = utils.denormalize_image_tensor(best_imgs)
        worst_imgs = torch.cat((originals[worst_idxs], recon[worst_idxs]), dim=0)
        # worst_imgs = utils.denormalize_image_tensor(worst_imgs)

        grid = torchvision.utils.make_grid(random_imgs, nrow=n_images)
        self.logger.add_image('random_reconstruction', grid, 0)

        grid = torchvision.utils.make_grid(best_imgs, nrow=n_images)
        self.logger.add_image('best_reconstruction', grid, 0)

        grid = torchvision.utils.make_grid(worst_imgs, nrow=n_images)
        self.logger.add_image('worst_reconstruction', grid, 0)
        
        print('test_loss:', avg_loss)

        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss}, tensorboard_logs

        return {'avg_test_loss': avg_loss}
    
    def test(self):
        _, test_metrics = self.test_loop(self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)

class GradCloner(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.clone_model = deepcopy(model)
        self.clone_optimizer = torch.optim.SGD(self.clone_model.parameters(), lr=0.)

    def copy_and_clear_grad(self):
        self.clone_optimizer.zero_grad()
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            cvalue.grad = pvalue.grad.clone()
        self.optimizer.zero_grad()

    def update_grad(self, alpha=1, beta=1):
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            cvalue.grad.data = \
                alpha * cvalue.grad.data + beta * pvalue.grad.data

class MaxMarginTrainer(AETrainer):
    def __init__(self, model, train_loader, val_loader, 
                    test_loader, optimizer, scheduler, 
                    device, args):
        super(MaxMarginTrainer, self).__init__(model, train_loader, 
                                                val_loader, test_loader, 
                                                optimizer, scheduler, 
                                                device, args)
        self.model_grad_dict = {n:None for n,p in self.model.named_parameters()}

    @classmethod
    def lp_dist_per_sample(cls, x, x_, p=2, reduction='sum'):
        diff = x - x_
        diff = diff.view(diff.shape[0], -1)
        
        if p == float('inf'):
            diff = torch.abs(diff)
            sm = nn.functional.gumbel_softmax(diff, dim=1)
            mse = (sm * diff).sum(1)
            # mse, _ = torch.max(diff, dim=1)
            return mse

        if p % 2 == 1:
            diff = torch.abs(diff)
        else:
            diff = diff ** p

        if reduction == 'sum':
            mse = (diff.sum(1))**(1/p)
        if reduction == 'mean':
            mse = ((diff.sum(1))**(1/p))/diff.shape[0]
        if reduction == 'none':
            mse = diff
        return mse

    def compute_reconstruction_loss(self, fx, x, fx_, x_, batch_size):
        if self.args.recon_weight > 0:
            reconstruction_loss = nn.functional.mse_loss(x, x_, reduction='none').view(x.shape[0],-1).mean(1)
            if self.args.use_feature_recon_loss:
                feature_loss = nn.functional.mse_loss(fx.detach(), fx_, reduction='none').view(x.shape[0],-1).mean(1)
                reconstruction_loss += self.args.feature_recon_weight * feature_loss                        

            if len(x) > batch_size:
                nAvgImgs = len(x) - batch_size
                avgSample_loss = reconstruction_loss[-nAvgImgs:]
                reconstruction_loss = reconstruction_loss[:-nAvgImgs]
                reconstruction_loss += avgSample_loss.sum()/len(reconstruction_loss)
            return reconstruction_loss
        else:
            return 0
    
    def compute_margin_loss(self, x, x_, p, correct, batch_size):
        if self.args.margin_weight > 0 and p is not None:
            x_ = x_[correct].detach()
            x = x[correct].detach()

            norm_p = self.args.norm_p if hasattr(self.args, 'norm_p') else float('inf')
            D = self.lp_dist_per_sample(x, x_, p=norm_p, reduction='sum').mean()
            proj_dist = self.lp_dist_per_sample(x, p, p=norm_p, reduction='sum')

            margin_loss = torch.zeros(batch_size, device=x.device)
            margin_loss[correct] = torch.relu(self.args.max_margin + D - proj_dist)

            return margin_loss
        else:
            return 0
    
    def compute_classification_loss(self, logits, y):
        classification_loss = 0
        if self.args.ce_weight > 0:
            if not hasattr(self.args, 'loss_type') or self.args.loss_type == 'xent':            
                if len(logits.shape) == 1 or logits.shape[1] == 1:
                    classification_loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1,1), y.view(-1,1), 
                                                                pos_weight=torch.tensor([9], device=y.device), 
                                                                reduction='none')                
                else:
                    classification_loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
            elif self.args.loss_type == 'hinge':            
                p_lengths = logits / torch.norm(self.classifier.weight, dim=1).view(1,-1)
                if len(logits.shape) == 1 or logits.shape[1] == 1:
                    y[y == 0] = -1
                    classification_loss = torch.relu(self.args.c - y*p_lengths)
                else:
                    classification_loss = nn.functional.multi_margin_loss(p_lengths, y, margin=self.args.c, reduction='none')        
            else:
                raise NotImplementedError
        return classification_loss

    def criterion(self, fx, x, fx_, x_, p, logits, y, p_logits, correct):
        reconstruction_loss = self.compute_reconstruction_loss(fx, x, fx_, x_, logits.shape[0])
        margin_loss = self.compute_margin_loss(x, x_, p, correct, logits.shape[0])
        classification_loss = self.compute_classification_loss(logits, y)
        proj_classification_loss = 0

        if self.args.ce_on_projections:
            proj_classification_loss = torch.zeros((logits.shape[0]), device=x.device)
            proj_classification_loss[correct] = torch.nn.functional.cross_entropy(p_logits, y[correct], reduction='none')
            proj_classification_loss *= self.args.proj_ce_weight

        loss = self.args.ce_weight * classification_loss + self.args.recon_weight * reconstruction_loss + self.args.margin_weight * margin_loss
        return reconstruction_loss, classification_loss, proj_classification_loss, margin_loss, loss
    
    def _update_grad_dict(self, loss, weight, excluded_modules=[], retain_graph=False):
        if isinstance(loss, torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
            
            exlcuded_module_names = [n for _,n in itertools.chain(*[m.named_parameters() for m in excluded_modules])]

            for n,p in self.model.named_parameters():
                if n not in exlcuded_module_names and p.grad is not None:
                    if self.model_grad_dict[n] is None:
                        self.model_grad_dict[n] = weight*p.grad.clone()
                    else:
                        self.model_grad_dict[n].data += weight*p.grad.data            
            self.optimizer.zero_grad()

    # def _optimization_wrapper(self, func):
    #     def wrapper(*args, **kwargs):            
    #         output, logs = func(*args, **kwargs)
    #         for n,p in self.model.named_parameters():
    #             p.grad = self.model_grad_dict[n]
    #         self.optimizer.step()
    #         return output, logs
    #     return wrapper

    def perform_update(self, reconstruction_loss, margin_loss, classification_loss):                
        self._update(reconstruction_loss, self.args.recon_weight, [self.model.classifier], retain_graph=True)        
        self._update(margin_loss, self.args.margin_weight, [self.model.encoder, self.model.decoder], retain_graph=True)        
        self._update(classification_loss, self.args.ce_weight, [self.model.decoder])

    def boundary_projections(self, fx, y, return_length_only=False):
        fx_shape = tuple(fx.shape)
        fx = fx.view(fx.shape[0], -1)

        W = self.model.classifier.weight
        W_ = W.unsqueeze(0).expand(W.shape[0], -1, -1)
        W_ = W.unsqueeze(1) - W_
        W = W_[y]
        W = W.transpose(1,2)
        W[torch.arange(W.shape[0]), :, y] = float('inf')
        W = W[torch.isfinite(W)].view(W.shape[0], W.shape[1], -1)

        norm = torch.norm(W, dim=1, keepdim=True)
        fx = fx.unsqueeze(1)
        p_lengths = torch.bmm(fx, W) / (norm + 1e-8)

        if return_length_only:
            return p_lengths.squeeze()

        P = p_lengths * W
        bias = self.model.classifier.bias
        b = torch.zeros((P.shape[0], bias.shape[0]), device=bias.device, requires_grad=True) + bias
        b[torch.arange(b.shape[0]), y] = float('inf')
        b = b[torch.isfinite(b)].view(b.shape[0], 1, -1)
        
        P += b
        fx = fx.transpose(1,2)
        fX_ = fx - P    
        
        outshape = fx_shape + (fX_.shape[2], )
        fX_ = fX_.view(*outshape)

        if not torch.isfinite(fX_).all():
            print('fX_:', fX_)
            exit(0)
        return fX_

    def get_closest_boundary_image(self, x, fP):
        if len(fP.shape) == 5:
            b, c, henc, wenc, nclasses = fP.shape
            fP = fP.permute(0,4,1,2,3).contiguous().view(-1, c, henc, wenc)
        if len(fP.shape) == 3:
            b, c, nclasses = fP.shape
            fP = fP.transpose(1,2).contiguous().view(-1, c)

        if hasattr(self.model, 'eval_decoder'):
            decoder = self.model.decoder
            self.model.decoder = self.model.eval_decoder
            P = self.model.decode(fP)
            self.model.decoder = decoder
        else:
            P = self.model.decode(fP)

        _, c, hdec, wdec = P.shape
        
        P = P.view(b * nclasses, -1)
        _x = x.view(-1, 1, c*hdec*wdec).expand(-1, nclasses, -1).contiguous().view(b * nclasses, -1)
        dist = self.lp_dist_per_sample(_x, P, p=self.args.norm_p, reduction='sum')
        dist = dist.view(b, nclasses)
        
        min_cls_idx = torch.argmin(dist, dim=1)
        dist = dist[range(dist.shape[0]), min_cls_idx]

        P = P.view(b, nclasses, c, hdec, wdec)        
        p = P[range(P.shape[0]), min_cls_idx]

        return p  
    
    def train(self):
        if self.args.pretrain_classifier_epochs > 0:
            self.model.freeze_ae()
        return super(MaxMarginTrainer, self).train()
        
    def train_step(self, batch, batch_idx):        
        if hasattr(self.model, 'eval_decoder') and batch_idx > 0 and batch_idx % self.args.decoder_update_freq == 0:
            self.model.update_decoder()

        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.args.num_averaged_training_images > 0:
            avg_pairs =self.get_averaged_images(x)
            x = torch.cat((x, avg_pairs), dim=0)

        fx, logits = self.model.forward(x)
        logits = logits[:y.shape[0]]

        # classification_loss = self.compute_classification_loss(logits, y)
        # self._update_grad_dict(classification_loss.mean(), self.args.ce_weight, excluded_modules=[self.model.decoder], retain_graph=True)

        fx_ = x_ = p = p_logits = None
        
        # pred = torch.argmax(logits, dim=1)
        # correct = torch.arange(y.shape[0])[y == pred]

        accuracy, correct = compute_accuracy(logits, y)
        correct = torch.arange(y.shape[0])[correct]
        
        if self.args.recon_weight > 0:
            x_ = self.model.decode(fx)
            if self.args.use_feature_recon_loss:
                fx_ = self.model.encode(x_)
                if not torch.isfinite(fx_).all():
                    exit(0)
            # reconstruction_loss = self.compute_reconstruction_loss(fx, x, fx_, x_, logits.shape[0])
            # self._update_grad_dict(reconstruction_loss.mean(), self.args.recon_weight, excluded_modules=[self.model.classifier])

        if self.args.margin_weight > 0 and len(correct) > 0:           
            fP = self.boundary_projections(fx[correct].detach(), y[correct]) # Correct x fdim x nclasses
            p = self.get_closest_boundary_image(x[correct], fP)

            # margin_loss = self.compute_margin_loss(x, x_, p, correct, logits.shape[0])
            # self._update_grad_dict(margin_loss.mean(), self.args.margin_weight, excluded_modules=[self.model.encoder, self.model.decoder])
            if self.args.ce_on_projections:
                _, p_logits = self.model.forward(p)
        
        
        reconstruction_loss, classification_loss, proj_classification_loss, margin_loss, loss = self.criterion(fx, x, fx_, x_, p, logits, y, p_logits, correct)
        # self.perform_update(reconstruction_loss.mean(), 
        #                     classification_loss.mean(), 
        #                     margin_loss.mean())

        metrics = {'train_reconstruction_loss': float(reconstruction_loss.mean() if isinstance(reconstruction_loss, torch.Tensor) else reconstruction_loss), 
                'train_classification_loss': float(classification_loss.mean() if isinstance(classification_loss, torch.Tensor) else classification_loss), 
                'train_proj_classification_loss':float(proj_classification_loss[correct].mean() if isinstance(proj_classification_loss, torch.Tensor) else proj_classification_loss),
                'train_margin_loss':float(margin_loss[correct].mean() if isinstance(margin_loss, torch.Tensor) else margin_loss),
                'train_loss':float(loss.mean()), 
                'train_accuracy':accuracy}

        return {"loss": loss.mean(), 'accuracy':accuracy}, metrics

    def train_epoch_end(self, outputs, metrics, epoch_idx):
        if hasattr(self.model, 'eval_decoder'):
            for ps, pt in zip(self.model.decoder.parameters(), self.model.eval_decoder.parameters()):
                pt.data = ps.data.clone()
        
        if self.args.pretrain_classifier_epochs - 1 <= epoch_idx:
            self.model.unfreeze_ae()

    def val_step(self, batch, batch_idx):        
        output, logs = self.train_step(batch, batch_idx)
        output['loss'] = output['loss'].detach().cpu()
        val_logs = {'lr':self.scheduler.optimizer.param_groups[0]['lr']}
        for k,v in logs.items():            
            val_logs[k.replace('train', 'val')] = v
        return output, val_logs

    def test_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        fx, logits = self.model.forward(x)

        pred = torch.argmax(logits, dim=1)
        correct = (y == pred)

        fx_ = x_ = p = p_logits = avg_pairs = torch.zeros(x.shape)

        if self.args.num_averaged_training_images > 0:
            avg_pairs = self.get_averaged_images(x)

        if self.args.recon_weight > 0:
            x_ = self.model.decode(fx)
            if self.args.use_feature_recon_loss:
                fx_ = self.model.encode(x_)

        if self.args.margin_weight > 0 and correct.any():            
            fP = self.boundary_projections(fx[correct], y[correct]) # Correct x fdim x nclasses
            p = torch.zeros_like(x_)            
            p[correct] = self.get_closest_boundary_image(x[correct], fP)            
            if self.args.ce_on_projections:
                _, p_logits = self.forward(p[correct])

        reconstruction_loss, classification_loss, proj_classification_loss, margin_loss, loss = self.criterion(fx, x, fx_, x_, p[correct], logits, y, p_logits , correct)
        loss = loss.detach().cpu()
        accuracy,_ = compute_accuracy(logits, y)
        logs = {                                                
            'test_accuracy':accuracy,
        }
        return {'test_loss':loss, 
                'test_accuracy':accuracy,
                'features': fx.detach().cpu(),
                'labels': y.detach().cpu(),
                'images': {'x': x.detach().cpu(), 
                            'x_':x_.detach().cpu(),
                            'p': p.detach().cpu(),
                            'avg_imgs': avg_pairs}
                }, logs
    def test_epoch_end(self, outputs, metrics):
        loss = torch.cat([x['test_loss'] for x in outputs], dim=0).cpu()
        
        originals = torch.cat([x['images']['x'] for x in outputs], dim=0)
        recon = torch.cat([x['images']['x_'] for x in outputs], dim=0)
        proj = torch.cat([x['images']['p'] for x in outputs], dim=0)
        avgd_imgs = torch.cat([x['images']['avg_imgs'] for x in outputs], dim=0)
        features = np.concatenate([x['features'].numpy() for x in outputs], axis=0)
        labels = np.concatenate([x['labels'].numpy() for x in outputs], axis=0)
        loss = loss.view(-1)
        
        feature_plot = plot_pca(features.reshape(features.shape[0], -1), labels)
        feature_plot = torchvision.transforms.ToTensor()(feature_plot)
        self.logger.add_image('PCA of Test Image Features', feature_plot, 0)

        avg_loss = float(loss.mean())
        avg_acc = np.mean([x['test_accuracy'] for x in outputs])

        n_images = 5
        loss_ = loss.clone()
        loss_[loss == 0] = loss.max()
        _, best_idxs = torch.topk(loss_, n_images, largest=False)        
        loss_ = loss.clone()
        loss_[loss == 0] = -1
        _, worst_idxs = torch.topk(loss_, n_images, largest=True)
        random_idxs = torch.randint(len(originals), (n_images,))

        best_imgs = torch.cat((originals[best_idxs], recon[best_idxs], proj[best_idxs]), dim=0)
        # best_imgs = utils.denormalize_image_tensor(best_imgs)
        worst_imgs = torch.cat((originals[worst_idxs], recon[worst_idxs], proj[worst_idxs]), dim=0)
        # worst_imgs = utils.denormalize_image_tensor(worst_imgs)
        random_imgs = torch.cat((originals[random_idxs], recon[random_idxs], proj[random_idxs]), dim=0)

        grid = torchvision.utils.make_grid(best_imgs, nrow=n_images)
        self.logger.add_image('best_reconstruction', grid, 0)

        grid = torchvision.utils.make_grid(worst_imgs, nrow=n_images)
        self.logger.add_image('worst_reconstruction', grid, 0)

        grid = torchvision.utils.make_grid(random_imgs, nrow=n_images)
        self.logger.add_image('random_reconstruction', grid, 0)

        grid = torchvision.utils.make_grid(avgd_imgs[np.random.randint(len(avgd_imgs), size=n_images)], nrow=n_images)

        robust_accuracy = 0
        
        print('test_loss:', avg_loss)
        print('test_accuracy:', avg_acc)        
        return {'avg_test_loss': avg_loss}, {'test_accuracy': avg_acc}