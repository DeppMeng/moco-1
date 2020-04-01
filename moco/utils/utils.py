class TSVDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """ Extends pytorch distributed sampler with in-process shuffling
    """

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = [i for i in indices if i % self.num_replicas == self.rank]
        else:
            indices = [i for i in range(len(sel.dataset)) if i % self.num_replicas == self.rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.num_samples - len(indices))]
        assert len(indices) == self.num_samples

        return iter(indices)