import torch
import copy
import inspect

COLLECT_STATS = False


def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if ref_labels is not None:
            ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, ref_labels


def check_shapes(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def add_to_recordable_attributes(input_obj, name=None, list_of_names=None, is_stat=False):
    if is_stat:
        attr_name_list_name = "_record_these_stats"
    else:
        attr_name_list_name = "_record_these"
    if not hasattr(input_obj, attr_name_list_name):
        setattr(input_obj, attr_name_list_name, [])
    attr_name_list = getattr(input_obj, attr_name_list_name)
    if name is not None:
        if name not in attr_name_list:
            attr_name_list.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)
    if list_of_names is not None and is_list_or_tuple(list_of_names):
        for n in list_of_names:
            add_to_recordable_attributes(input_obj, name=n, is_stat=is_stat)


def reset_stats(input_obj):
    for attr_list in ["_record_these_stats"]:
        for r in getattr(input_obj, attr_list, []):
            setattr(input_obj, r, 0)


def labels_or_indices_tuple_required(labels, indices_tuple):
    if labels is None and indices_tuple is None:
        raise ValueError("labels and indices_tuple cannot both be None")


def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs


def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


def convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return get_random_triplet_indices(
                labels, ref_labels, t_per_anchor=t_per_anchor
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]


def get_random_triplet_indices(labels, ref_labels=None, t_per_anchor=None, weights=None):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = torch.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
        n_inds = torch.where(ref_labels != label)[0]
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = to_device(torch.cat(a_idx), device=labels_device, dtype=torch.long)
        p_idx = to_device(torch.cat(p_idx), device=labels_device, dtype=torch.long)
        n_idx = to_device(torch.cat(n_idx), device=labels_device, dtype=torch.long)
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.tensor([], device=labels_device, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone()


def meshgrid_from_sizes(x, y, dim=0):
    a = torch.arange(x.size(dim), device=x.device)
    b = torch.arange(y.size(dim), device=y.device)
    return torch.meshgrid(a, b, indexing="ij")


class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = (
            COLLECT_STATS if collect_stats is None else collect_stats
        )

    def add_to_recordable_attributes(
            self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )

    def reset_stats(self):
        reset_stats(self)


class BaseReducer(ModuleWithRecords):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_to_recordable_attributes(name="losses_size", is_stat=True)

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
            self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)


class DoNothingReducer(BaseReducer):
    def forward(self, loss_dict, embeddings, labels):
        return loss_dict


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)


class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(
            len(loss_dict), dtype=embeddings.dtype, device=embeddings.device
        )
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)


class DivisorReducer(BaseReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_to_recordable_attributes(name="divisor", is_stat=True)

    def unpack_loss_info(self, loss_info):
        losses, loss_indices, reduction_type, kwargs = super().unpack_loss_info(
            loss_info
        )
        if reduction_type != "already_reduced":
            kwargs = {"divisor": loss_info["divisor"]}
            self.divisor = kwargs["divisor"]
        return losses, loss_indices, reduction_type, kwargs

    def sum_and_divide(self, losses, embeddings, divisor):
        if divisor != 0:
            output = torch.sum(losses) / divisor
            if torch.isnan(output) and losses.dtype == torch.float16:
                output = torch.sum(to_dtype(losses, dtype=torch.float32)) / divisor
                output = to_dtype(output, dtype=torch.float16)
            return output
        return self.zero_loss(embeddings)

    def element_reduction(self, losses, loss_indices, embeddings, labels, divisor=1):
        return self.sum_and_divide(losses, embeddings, divisor)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def triplet_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)


class BaseDistance(ModuleWithRecords):
    def __init__(
            self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=[
                "initial_avg_query_norm",
                "initial_avg_ref_norm",
                "final_avg_query_norm",
                "final_avg_ref_norm",
            ],
            is_stat=True,
        )

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        self.check_shapes(query_emb, ref_emb)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(
            self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            with torch.no_grad():
                self.initial_avg_query_norm = torch.mean(
                    self.get_norm(query_emb)
                ).item()
                self.initial_avg_ref_norm = torch.mean(self.get_norm(ref_emb)).item()
                self.final_avg_query_norm = torch.mean(
                    self.get_norm(query_emb_normalized)
                ).item()
                self.final_avg_ref_norm = torch.mean(
                    self.get_norm(ref_emb_normalized)
                ).item()

    def check_shapes(self, query_emb, ref_emb):
        if query_emb.ndim != 2 or (ref_emb is not None and ref_emb.ndim != 2):
            raise ValueError(
                "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
            )


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class EmbeddingRegularizerMixin:
    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        self.embedding_regularizer = (
                embedding_regularizer is not None
        )  # hack needed to know whether reg will be in sub-loss names
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight
        if self.embedding_regularizer is not None:
            self.add_to_recordable_attributes(
                list_of_names=["embedding_reg_weight"], is_stat=False
            )

    def embedding_regularization_loss(self, embeddings):
        if self.embedding_regularizer is None:
            loss = 0
        else:
            loss = self.embedding_regularizer(embeddings) * self.embedding_reg_weight
        return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        if self.embedding_regularizer is not None:
            loss_dict["embedding_reg_loss"] = self.embedding_regularization_loss(
                embeddings
            )

    def regularization_loss_names(self):
        return ["embedding_reg_loss"]


class ModuleWithRecordsAndReducer(ModuleWithRecords):
    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_reducer(reducer)

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        if isinstance(reducer, (MultipleReducers, DoNothingReducer)):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = (
                self.get_default_reducer()
                if reducer is None
                else copy.deepcopy(reducer)
            )
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = (
                    self.get_default_reducer()
                    if reducer is None
                    else copy.deepcopy(reducer)
                )
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ["loss"]


class ModuleWithRecordsAndDistance(ModuleWithRecords):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2)


class ModuleWithRecordsReducerAndDistance(ModuleWithRecordsAndReducer, ModuleWithRecordsAndDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseMetricLossFunction(EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(
            self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        check_shapes(embeddings, labels)
        if labels is not None:
            labels = to_device(labels, embeddings)
        ref_emb, ref_labels = set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = "RegularizerMixin"
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, "").lower()
                if getattr(self, "{}_regularizer".format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names


class MarginLoss(BaseMetricLossFunction):
    def __init__(
            self,
            margin=0.2,
            nu=0,
            beta=1.2,
            triplets_per_anchor="all",
            learn_beta=False,
            num_classes=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.nu = nu
        self.learn_beta = learn_beta
        self.initialize_beta(beta, num_classes)
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(
            list_of_names=["margin", "nu", "beta"], is_stat=False
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = convert_to_triplets(
            indices_tuple, labels, ref_labels, self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()

        beta = self.beta if len(self.beta) == 1 else self.beta[labels[anchor_idx]]
        beta = to_device(beta, device=embeddings.device, dtype=embeddings.dtype)

        mat = self.distance(embeddings, ref_emb)

        d_ap = mat[anchor_idx, positive_idx]
        d_an = mat[anchor_idx, negative_idx]

        pos_loss = torch.nn.functional.relu(
            self.distance.margin(d_ap, beta) + self.margin
        )
        neg_loss = torch.nn.functional.relu(
            self.distance.margin(beta, d_an) + self.margin
        )

        num_pos_pairs = torch.sum(pos_loss > 0.0)
        num_neg_pairs = torch.sum(neg_loss > 0.0)

        divisor = num_pos_pairs + num_neg_pairs

        margin_loss = pos_loss + neg_loss

        loss_dict = {
            "margin_loss": {
                "losses": margin_loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
                "divisor": divisor,
            },
            "beta_reg_loss": self.compute_reg_loss(beta, anchor_idx, divisor),
        }

        return loss_dict

    def compute_reg_loss(self, beta, anchor_idx, divisor):
        if self.learn_beta:
            loss = beta * self.nu
            if len(self.beta) == 1:
                return {
                    "losses": loss,
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            else:
                return {
                    "losses": loss,
                    "indices": anchor_idx,
                    "reduction_type": "element",
                    "divisor": divisor,
                }
        return self.zero_loss()

    def _sub_loss_names(self):
        return ["margin_loss", "beta_reg_loss"]

    def get_default_reducer(self):
        return DivisorReducer()

    def initialize_beta(self, beta, num_classes):
        self.beta = torch.tensor([float(beta)])
        if num_classes:
            self.beta = torch.ones(num_classes) * self.beta
        if self.learn_beta:
            self.beta = torch.nn.Parameter(self.beta)
