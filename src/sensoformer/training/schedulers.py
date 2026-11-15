from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

class LRSched:
    @staticmethod
    def plateau(optimizer, args):
        """
        ReduceLROnPlateau on val loss.
        """
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,          
            patience=args.plateau_wait, 
            threshold=1e-5,
        )
        return scheduler

    @staticmethod
    def warmup_cosine(optimizer, args):
        """
        Linear warmup for a few optimizer steps, then cosine decay
        for the remaining optimizer steps.
        """
        total_steps  = max(1, int(args.total_train_steps))
        warmup_steps = max(0, int(args.warmup_steps))  
        cosine_steps = max(1, total_steps - warmup_steps)

        # extra args with safe defaults
        min_lr = 1e-8
        warm_start_factor = 1e-4

        # no warmup case
        if warmup_steps == 0:
            return CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=min_lr,
            )

        # 1) linear warmup from start_factor * lr to 1.0 * lr
        sched_warmup = LinearLR(
            optimizer,
            start_factor=warm_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # 2) cosine decay from lr to min_lr
        sched_cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_lr,
        )

        # 3) chain them together
        scheduler = SequentialLR(
            optimizer,
            schedulers=[sched_warmup, sched_cosine],
            milestones=[warmup_steps],  
        )
        return scheduler
    
    @staticmethod
    def cosine_decay(optimizer, total_train_steps = 100, min_lr=1e-9):
        """
        Cosine decay from initial lr to min_lr over total optimizer steps.
        Call scheduler.step() after every optimizer.step().
        """
        total_steps = max(1, int(total_train_steps))

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )
        return scheduler
        
