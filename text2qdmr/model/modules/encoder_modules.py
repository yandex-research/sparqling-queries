import itertools

import numpy as np
import torch

from text2qdmr.model.modules import transformer
from text2qdmr.utils import batched_sequence


def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def get_attn_mask(seq_lengths):
    # Given seq_lengths like [3, 1, 2], this will produce
    # [[[1, 1, 1],
    #   [1, 1, 1],
    #   [1, 1, 1]],
    #  [[1, 0, 0],
    #   [0, 0, 0],
    #   [0, 0, 0]],
    #  [[1, 1, 0],
    #   [1, 1, 0],
    #   [0, 0, 0]]]
    # int(max(...)) so that it has type 'int instead of numpy.int64
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask

class RelationalTransformerUpdate(torch.nn.Module):

    class RelationMap():
        def __init__(self,
                    merge_types=False,
                    tie_layers=False,
                    qq_max_dist=2,
                    # qc_token_match=True,
                    # qt_token_match=True,
                    # cq_token_match=True,
                    cc_foreign_key=True,
                    cc_table_match=True,
                    cc_max_dist=2,
                    ct_foreign_key=True,
                    ct_table_match=True,
                    # tq_token_match=True,
                    tc_table_match=True,
                    tc_foreign_key=True,
                    tt_max_dist=2,
                    tt_foreign_key=True,
                    qv_token_match=False,
                    qv_default=False,
                    vv_max_dist=2,
                    sc_link=False,
                    cv_link=False,
                    full_grnd_link=False,
                    merge_sc_link=False,
                    default_graph_link=False,
                    default_link_and_graph=False):

            self.qq_max_dist = qq_max_dist
            # self.qc_token_match = qc_token_match
            # self.qt_token_match = qt_token_match
            # self.cq_token_match = cq_token_match
            self.cc_foreign_key = cc_foreign_key
            self.cc_table_match = cc_table_match
            self.cc_max_dist = cc_max_dist
            self.ct_foreign_key = ct_foreign_key
            self.ct_table_match = ct_table_match
            # self.tq_token_match = tq_token_match
            self.tc_table_match = tc_table_match
            self.tc_foreign_key = tc_foreign_key
            self.tt_max_dist = tt_max_dist
            self.tt_foreign_key = tt_foreign_key
            self.qv_token_match = qv_token_match
            self.vv_max_dist = vv_max_dist

            self.full_grnd_link = full_grnd_link
            self.merge_sc_link = merge_sc_link
            self.default_graph_link = default_graph_link
            self.default_link_and_graph = default_link_and_graph

            self.relation_ids = {}

            def add_relation(name):
                self.relation_ids[name] = len(self.relation_ids)

            def add_rel_dist(name, max_dist):
                for i in range(-max_dist, max_dist + 1):
                    add_relation((name, i))

            add_rel_dist('qq_dist', qq_max_dist)

            add_relation('qc_default')
            # if qc_token_match:
            #    add_relation('qc_token_match')

            add_relation('qt_default')
            # if qt_token_match:
            #    add_relation('qt_token_match')

            add_relation('cq_default')
            # if cq_token_match:
            #    add_relation('cq_token_match')

            if qv_token_match:
                add_relation('qvVEM')
                add_relation('vqVEM')
                add_relation('qvVPM')
                add_relation('vqVPM')
                add_relation("cvCELLMATCH")
                add_relation("vcCELLMATCH")

            if qv_default:
                add_relation('qv_default')
                add_relation('vq_default')
                add_relation('tv_default')
                add_relation('vt_default')
                add_relation('cv_default')
                add_relation('vc_default')
                add_relation('vv_default')
                add_rel_dist('vv_dist', vv_max_dist)

            add_relation('cc_default')

            if default_graph_link:
                add_relation('graph_default')

            if cc_foreign_key:
                add_relation('cc_foreign_key_forward')
                add_relation('cc_foreign_key_backward')
            if cc_table_match:
                add_relation('cc_table_match')
            add_rel_dist('cc_dist', cc_max_dist)

            add_relation('ct_default')
            if ct_foreign_key:
                add_relation('ct_foreign_key')
            if ct_table_match:
                add_relation('ct_primary_key')
                add_relation('ct_table_match')
                add_relation('ct_any_table')

            add_relation('tq_default')
            # if cq_token_match:
            #    add_relation('tq_token_match')

            add_relation('tc_default')
            if tc_table_match:
                add_relation('tc_primary_key')
                add_relation('tc_table_match')
                add_relation('tc_any_table')
            if tc_foreign_key:
                add_relation('tc_foreign_key')

            add_relation('tt_default')
            if tt_foreign_key:
                add_relation('tt_foreign_key_forward')
                add_relation('tt_foreign_key_backward')
                add_relation('tt_foreign_key_both')
            add_rel_dist('tt_dist', tt_max_dist)

            # schema linking relations
            # forward_backward
            if sc_link:
                add_relation('qcCEM')
                add_relation('cqCEM')
                add_relation('qcCPM')
                add_relation('cqCPM')
                if not merge_sc_link:
                    add_relation('qtTEM')
                    add_relation('tqTEM')
                    add_relation('qtTPM')
                    add_relation('tqTPM')

            if cv_link:
                add_relation("qcNUMBER")
                add_relation("cqNUMBER")
                add_relation("qcTIME")
                add_relation("cqTIME")
                add_relation("qcCELLMATCH")
                add_relation("cqCELLMATCH")

            if merge_types:
                assert not cc_foreign_key
                assert not cc_table_match
                assert not ct_foreign_key
                assert not ct_table_match
                assert not tc_foreign_key
                assert not tc_table_match
                assert not tt_foreign_key

                assert cc_max_dist == qq_max_dist
                assert tt_max_dist == qq_max_dist

                add_relation('xx_default')
                self.relation_ids['qc_default'] = self.relation_ids['xx_default']
                self.relation_ids['qt_default'] = self.relation_ids['xx_default']
                self.relation_ids['cq_default'] = self.relation_ids['xx_default']
                self.relation_ids['cc_default'] = self.relation_ids['xx_default']
                self.relation_ids['ct_default'] = self.relation_ids['xx_default']
                self.relation_ids['tq_default'] = self.relation_ids['xx_default']
                self.relation_ids['tc_default'] = self.relation_ids['xx_default']
                self.relation_ids['tt_default'] = self.relation_ids['xx_default']

                if sc_link:
                    self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
                    self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
                    self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
                    self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
                    self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
                    self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
                    self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
                    self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
                if cv_link:
                    self.relation_ids["qcNUMBER"] = self.relation_ids['xx_default']
                    self.relation_ids["cqNUMBER"] = self.relation_ids['xx_default']
                    self.relation_ids["qcTIME"] = self.relation_ids['xx_default']
                    self.relation_ids["cqTIME"] = self.relation_ids['xx_default']
                    self.relation_ids["qcCELLMATCH"] = self.relation_ids['xx_default']
                    self.relation_ids["cqCELLMATCH"] = self.relation_ids['xx_default']

                for i in range(-qq_max_dist, qq_max_dist + 1):
                    self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist', i]
                    self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist', i]



    def __init__(self, device, num_layers, num_heads, hidden_size,
                 ff_size=None,
                 dropout=0.1,
                 merge_types=False,
                 tie_layers=False,
                 qq_max_dist=2,
                 # qc_token_match=True,
                 # qt_token_match=True,
                 # cq_token_match=True,
                 cc_foreign_key=True,
                 cc_table_match=True,
                 cc_max_dist=2,
                 ct_foreign_key=True,
                 ct_table_match=True,
                 # tq_token_match=True,
                 tc_table_match=True,
                 tc_foreign_key=True,
                 tt_max_dist=2,
                 tt_foreign_key=True,
                 qv_token_match=False,
                 qv_default=False,
                 vv_max_dist=2,
                 sc_link=False,
                 cv_link=False,
                 full_grnd_link=False,
                 merge_sc_link=False,
                 default_graph_link=False,
                 default_link_and_graph=False
                 ):
        super().__init__()
        self._device = device
        self.num_heads = num_heads

        self.relation_map = self.RelationMap(
                 merge_types=merge_types,
                 tie_layers=tie_layers,
                 qq_max_dist=qq_max_dist,
                 # qc_token_match=qc_token_match,
                 # qt_token_match=qt_token_match,
                 # cq_token_match=cq_token_match,
                 cc_foreign_key=cc_foreign_key,
                 cc_table_match=cc_table_match,
                 cc_max_dist=cc_max_dist,
                 ct_foreign_key=ct_foreign_key,
                 ct_table_match=ct_table_match,
                 # tq_token_match=tq_token_match,
                 tc_table_match=tc_table_match,
                 tc_foreign_key=tc_foreign_key,
                 tt_max_dist=tt_max_dist,
                 tt_foreign_key=tt_foreign_key,
                 qv_token_match=qv_token_match,
                 qv_default=qv_default,
                 vv_max_dist=vv_max_dist,
                 sc_link=sc_link,
                 cv_link=cv_link,
                 full_grnd_link=full_grnd_link,
                 merge_sc_link=merge_sc_link,
                 default_graph_link=default_graph_link,
                 default_link_and_graph=default_link_and_graph)

        if ff_size is None:
            ff_size = hidden_size * 4
        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout),
                transformer.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                len(self.relation_map.relation_ids),
                dropout),
            hidden_size,
            num_layers,
            tie_layers)

        self.align_attn = transformer.PointerWithRelations(hidden_size,
                                                           len(self.relation_map.relation_ids), dropout)

    def create_align_mask(self, num_head, q_length, c_length, t_length):
        # mask with size num_heads * all_len * all * len
        all_length = q_length + c_length + t_length
        mask_1 = torch.ones(num_head - 1, all_length, all_length, device=self._device)
        mask_2 = torch.zeros(1, all_length, all_length, device=self._device)
        for i in range(q_length):
            for j in range(q_length, q_length + c_length):
                mask_2[0, i, j] = 1
                mask_2[0, j, i] = 1
        mask = torch.cat([mask_1, mask_2], 0)
        return mask


    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc=None, t_boundaries=None,
                            v_enc=None, v_boundaries=None, use_relations=False, input_types=None):
        # returns:
        #    q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat)
        # alternative call:
        #    forward_unbatched(self, desc, q_enc, grnd_enc, grnd_boundaries)
        # in this case it will return same entities for columns and tables:
        #    q_enc_new, grnd_enc_new, grnd_enc_new, (m2grnd_align_mat, m2grnd_align_mat)

        if t_enc is not None:
            assert t_boundaries is not None
            if v_enc is None:
                # enc shape: total len x batch (=1) x recurrent size
                enc = torch.cat((q_enc, c_enc, t_enc), dim=0)
            else:
                # enc shape: total len x batch (=1) x recurrent size
                enc = torch.cat((q_enc, c_enc, t_enc, v_enc), dim=0)

            # enc shape: batch (=1) x total len x recurrent size
            enc = enc.transpose(0, 1)

            # Catalogue which things are where
            relations = self.compute_relations(
                desc,
                enc_length=enc.shape[1],
                q_enc_length=q_enc.shape[0],
                c_enc_length=c_enc.shape[0],
                t_enc_length=t_enc.shape[0],
                c_boundaries=c_boundaries,
                t_boundaries=t_boundaries,
                v_boundaries=v_boundaries,
                relation_map=self.relation_map)

            relations_t = torch.LongTensor(relations).to(self._device)
            enc_new = self.encoder(enc, relations_t, mask=None)

            # Split updated_enc again
            c_base = q_enc.shape[0]
            t_base = q_enc.shape[0] + c_enc.shape[0]
            v_base = q_enc.shape[0] + c_enc.shape[0] + t_enc.shape[0]
            if v_enc is None:
                assert v_base == enc_new.shape[1]
            q_enc_new = enc_new[:, :c_base]
            c_enc_new = enc_new[:, c_base:t_base]
            t_enc_new = enc_new[:, t_base:v_base]
            v_enc_new = enc_new[:, v_base:]

            m2c_align_mat = self.align_attn(enc_new, enc_new[:, c_base:t_base], \
                                            enc_new[:, c_base:t_base], relations_t[:, c_base:t_base])
            m2t_align_mat = self.align_attn(enc_new, enc_new[:, t_base:v_base], \
                                            enc_new[:, t_base:v_base], relations_t[:, t_base:v_base])
            if v_enc is not None:
                m2v_align_mat = self.align_attn(enc_new, enc_new[:, v_base:], \
                                            enc_new[:, v_base:], relations_t[:, v_base:])
            else:
                m2v_align_mat = None
            return q_enc_new, c_enc_new, t_enc_new, v_enc_new, (m2c_align_mat, m2t_align_mat, m2v_align_mat)
        else:
            # assert t_boundaries is None
            grnd_enc, grnd_boundaries = c_enc, c_boundaries

            # enc shape: total len x batch (=1) x recurrent size
            enc = torch.cat((q_enc, grnd_enc), dim=0)

            # enc shape: batch (=1) x total len x recurrent size
            enc = enc.transpose(0, 1)

            # Catalogue which things are where
            total_len = enc.shape[1]
            if use_relations:
                # Catalogue which things are where
                relations = self.compute_relations(
                    desc,
                    enc_length=enc.shape[1],
                    q_enc_length=q_enc.shape[0],
                    c_enc_length=grnd_enc.shape[0],
                    t_enc_length=None,
                    c_boundaries=grnd_boundaries,
                    t_boundaries=None,
                    input_types=input_types,
                    v_boundaries=None,
                    relation_map=self.relation_map)
            else:
                relations = np.zeros((total_len, total_len), dtype=np.int64)

            relations_t = torch.LongTensor(relations).to(self._device)
            enc_new = self.encoder(enc, relations_t, mask=None)

            # Split updated_enc again
            grnd_base = q_enc.shape[0]
            q_enc_new = enc_new[:, :grnd_base]
            grnd_enc_new = enc_new[:, grnd_base:]

            m2grnd_align_mat = self.align_attn(enc_new, enc_new[:, grnd_base:], \
                                               enc_new[:, grnd_base:], relations_t[:, grnd_base:])

            return q_enc_new, grnd_enc_new, grnd_enc_new, (m2grnd_align_mat, m2grnd_align_mat)


    def forward_batched(self, relations_batch, q_enc_batch, grnd_enc_batch):
        # returns:
        #    q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat)
        # alternative call:
        #    forward_unbatched(self, desc, q_enc, grnd_enc, grnd_boundaries)
        # in this case it will return same entities for columns and tables:
        #    q_enc_new, grnd_enc_new, grnd_enc_new, (m2grnd_align_mat, m2grnd_align_mat)

        # merge encodings from batch elements
        q_enc_lens_batch = [enc.shape[0] for enc in q_enc_batch]
        grnd_enc_lens_batch = [enc.shape[0] for enc in grnd_enc_batch]
        total_lens_batch = [q_len + grnd_len for q_len, grnd_len in zip(q_enc_lens_batch, grnd_enc_lens_batch)]

        enc_batch = [torch.cat([q_enc, grnd_enc], dim=0) for q_enc, grnd_enc in zip(q_enc_batch, grnd_enc_batch)]
        enc_batch = torch.nn.utils.rnn.pad_sequence(enc_batch, batch_first=True, padding_value=0)

        # merge relations from batch elements
        batch_size = len(relations_batch)
        relations_max_size = [max(relations.size(i_dim) for relations in relations_batch) for i_dim in range(2)]
        relations_batch_th = torch.zeros([batch_size] + list(relations_max_size),
                                            device=relations_batch[0].device,
                                            dtype=relations_batch[0].dtype)
        for i_b in range(batch_size):
            relations_batch_th[i_b,
                                :relations_batch[i_b].size(0),
                                :relations_batch[i_b].size(1)] =\
                                    relations_batch[i_b]
        relations_batch_th = relations_batch_th.to(device=enc_batch.device)

        # create mask: [batch, num queries, num kv]
        mask_batch = torch.arange(max(total_lens_batch))[None, :] < torch.tensor(total_lens_batch)[:, None]
        mask_batch = mask_batch.to(device=enc_batch.device)
        mask_batch = mask_batch.unsqueeze(1) # prepare dimensions for masked attention

        # run transformer
        enc_batch_new = self.encoder(enc_batch, relations_batch_th, mask=mask_batch)

        # Split updated_enc again
        q_enc_new = [enc[:q_enc_len] for enc, q_enc_len in zip(enc_batch_new, q_enc_lens_batch)]
        grnd_enc_new = [enc[q_enc_len:q_enc_len+grnd_enc_len] for enc, q_enc_len, grnd_enc_len in zip(enc_batch_new, q_enc_lens_batch, grnd_enc_lens_batch)]

        # q_enc_new = torch.nn.utils.rnn.pad_sequence(q_enc_new, batch_first=True, padding_value=0)
        # grnd_enc_new = torch.nn.utils.rnn.pad_sequence(grnd_enc_new, batch_first=True, padding_value=0)

        m2grnd_align_mat_batch = []
        for i_b in range(batch_size):
            # forward(self, query, key, value, relation, mask=None)
            grnd_start = q_enc_lens_batch[i_b]
            seq_len = total_lens_batch[i_b]
            m2grnd_align_mat = self.align_attn(enc_batch_new[i_b][None, :seq_len], enc_batch_new[i_b][None, grnd_start:seq_len], \
                                                enc_batch_new[i_b][None, grnd_start:seq_len], relations_batch_th[i_b][:seq_len, grnd_start:seq_len])
            m2grnd_align_mat_batch.append(m2grnd_align_mat)

        return q_enc_new, grnd_enc_new, grnd_enc_new, (m2grnd_align_mat_batch, m2grnd_align_mat_batch)

    @classmethod
    def compute_relations(cls, desc, enc_length, q_enc_length, c_enc_length, t_enc_length,
                            c_boundaries, t_boundaries, v_boundaries,
                            relation_map,
                            input_types=None):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}, 'q_val_match': {}, 'col_val_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})
        # print(sc_link)

        # Catalogue which things are where
        loc_types = {}
        for i in range(q_enc_length):
            loc_types[i] = ('question',)
        if not input_types:
            c_base = q_enc_length
            for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
                for i in range(c_start + c_base, c_end + c_base):
                    loc_types[i] = ('column', c_id)

            if t_boundaries is not None:
                t_base = q_enc_length + c_enc_length
                for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
                    for i in range(t_start + t_base, t_end + t_base):
                        loc_types[i] = ('table', t_id)
            if v_boundaries is not None:
                v_base = q_enc_length + c_enc_length + t_enc_length
                for v_id, (v_start, v_end) in enumerate(zip(v_boundaries, v_boundaries[1:])):
                    for i in range(v_start + v_base, v_end + v_base):
                        loc_types[i] = ('value', v_id)
        else:
            base = q_enc_length

            tab_len = input_types.get('table', 0)
            col_len = tab_len + input_types.get('column', 0)
            val_len = col_len + input_types.get('value', 0)
            assert len(c_boundaries) == val_len + 1, (c_boundaries, input_types)

            for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
                for i in range(c_start + base, c_end + base):
                    if c_id < tab_len:
                        loc_types[i] = ('table', c_id)
                    elif c_id >= tab_len and c_id < col_len:
                        loc_types[i] = ('column', c_id)
                    elif c_id >= col_len:
                        loc_types[i] = ('value', c_id)
            t_base = base
            c_base = base
            v_base = base

        relations = np.zeros((enc_length, enc_length), dtype=np.int64)

        for i, j in itertools.product(range(enc_length), repeat=2):
            def set_relation(name):
                relations[i, j] = relation_map.relation_ids[name]

            i_type, j_type = loc_types[i], loc_types[j]
            if i_type[0] == 'question':
                if j_type[0] == 'question':
                    set_relation(('qq_dist', clamp(j - i, relation_map.qq_max_dist)))
                elif relation_map.default_link_and_graph:
                    set_relation('qc_default')
                    if relation_map.merge_sc_link:
                        j_real = j - base
                        if f"{i},{j_real}" in sc_link["q_col_match"]:
                            set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
                elif j_type[0] == 'column':
                    # set_relation('qc_default')
                    j_real = j - c_base
                    if f"{i},{j_real}" in sc_link["q_col_match"]:
                        set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["cell_match"]:
                        set_relation("qc" + cv_link["cell_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["num_date_match"]:
                        set_relation("qc" + cv_link["num_date_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qc_default')

                    if relation_map.full_grnd_link and f"{i},{j_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])
                    if relation_map.full_grnd_link and f"{i},{j_real}" in sc_link["q_val_match"]:
                        set_relation("qv" + sc_link["q_val_match"][f"{i},{j_real}"])

                elif j_type[0] == 'table':
                    # set_relation('qt_default')
                    j_real = j - t_base
                    if f"{i},{j_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qt_default')
                elif j_type[0] == 'value':
                    j_real = j - v_base
                    assert j_real >= 0, j_real
                    if f"{i},{j_real}" in sc_link["q_val_match"]:
                        set_relation("qv" + sc_link["q_val_match"][f"{i},{j_real}"])
                    else:
                        set_relation('qv_default')

            elif i_type[0] == 'column':
                if j_type[0] == 'question':
                    # set_relation('cq_default')
                    i_real = i - c_base
                    if f"{j},{i_real}" in sc_link["q_col_match"]:
                        set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["cell_match"]:
                        set_relation("cq" + cv_link["cell_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["num_date_match"]:
                        set_relation("cq" + cv_link["num_date_match"][f"{j},{i_real}"])
                    else:
                        set_relation('cq_default')

                    if relation_map.full_grnd_link and f"{j},{i_real}" in sc_link["q_tab_match"]:
                        set_relation("tq" + sc_link["q_tab_match"][f"{j},{i_real}"])
                    if relation_map.full_grnd_link and f"{j},{i_real}" in sc_link["q_val_match"]:
                        set_relation("vq" + sc_link["q_val_match"][f"{j},{i_real}"])

                elif relation_map.default_link_and_graph:
                    if j_type[0] == 'column':
                        col1, col2 = i_type[1], j_type[1]
                        if col1 == col2:
                            set_relation(('cc_dist', clamp(j - i, relation_map.cc_max_dist)))
                        else:
                            set_relation('cc_default')
                            if relation_map.cc_foreign_key:
                                if desc['foreign_keys'] and desc['foreign_keys'].get(str(col1)) == col2:
                                    if relation_map.default_graph_link:
                                        set_relation('graph_default')
                                    else:
                                        set_relation('cc_foreign_key_forward')
                                if desc['foreign_keys'] and desc['foreign_keys'].get(str(col2)) == col1:
                                    if relation_map.default_graph_link:
                                        set_relation('graph_default')
                                    else:
                                        set_relation('cc_foreign_key_backward')
                            if (relation_map.cc_table_match and desc['column_to_table'] and
                                    desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
                                if relation_map.default_graph_link:
                                    set_relation('graph_default')
                                else:
                                    set_relation('cc_table_match')
                    elif j_type[0] == 'table':
                        col, table = i_type[1], j_type[1]
                        set_relation('cc_default')
                        if relation_map.ct_foreign_key and cls.match_foreign_key(desc, col, table):
                            if relation_map.default_graph_link:
                                set_relation('graph_default')
                            else:
                                set_relation('ct_foreign_key')
                        if relation_map.ct_table_match:
                            col_table = desc['column_to_table'][str(col)]
                            if col_table == table:
                                if relation_map.default_graph_link:
                                    set_relation('graph_default')
                                elif col in desc['primary_keys']:
                                    set_relation('ct_primary_key')
                                else:
                                    set_relation('ct_table_match')
                            elif col_table is None:
                                raise RuntimeError
                                set_relation('ct_any_table')
                    elif j_type[0] == 'value':
                        set_relation('cc_default')
                        i_real = i - base
                        j_real = j - base
                        assert i_real >= 0 and j_real >= 0, (i_real, j_real)
                        if f"{i_real},{j_real}" in sc_link["col_val_match"]:
                            set_relation("cv" + sc_link["col_val_match"][f"{i_real},{j_real}"])

                elif j_type[0] == 'column':
                    col1, col2 = i_type[1], j_type[1]
                    if col1 == col2:
                        set_relation(('cc_dist', clamp(j - i, relation_map.cc_max_dist)))
                    else:
                        set_relation('cc_default')
                        if relation_map.cc_foreign_key:
                            if desc['foreign_keys'] and desc['foreign_keys'].get(str(col1)) == col2:
                                set_relation('cc_foreign_key_forward')
                            if desc['foreign_keys'] and desc['foreign_keys'].get(str(col2)) == col1:
                                set_relation('cc_foreign_key_backward')
                        if (relation_map.cc_table_match and desc['column_to_table'] and
                                desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
                            set_relation('cc_table_match')

                elif j_type[0] == 'table':
                    col, table = i_type[1], j_type[1]
                    set_relation('ct_default')
                    if relation_map.ct_foreign_key and cls.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    if relation_map.ct_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('ct_primary_key')
                            else:
                                set_relation('ct_table_match')
                        elif col_table is None:
                            set_relation('ct_any_table')

                elif j_type[0] == 'value':
                    i_real = i - c_base
                    j_real = j - v_base
                    assert i_real >= 0 and j_real >= 0, (i_real, j_real)
                    if f"{i_real},{j_real}" in sc_link["col_val_match"]:
                        set_relation("cv" + sc_link["col_val_match"][f"{i_real},{j_real}"])
                    else:
                        set_relation('cv_default')

            elif i_type[0] == 'table':
                if j_type[0] == 'question':
                    # set_relation('tq_default')
                    i_real = i - t_base
                    if f"{j},{i_real}" in sc_link["q_tab_match"]:
                        set_relation("tq" + sc_link["q_tab_match"][f"{j},{i_real}"])
                    elif relation_map.default_link_and_graph:
                        set_relation('cq_default')
                        if relation_map.merge_sc_link:
                            if f"{j},{i_real}" in sc_link["q_col_match"]:
                                set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
                    else:
                        set_relation('tq_default')
                elif relation_map.default_link_and_graph:
                    if j_type[0] == 'column':
                        table, col = i_type[1], j_type[1]
                        set_relation('cc_default')

                        if relation_map.tc_foreign_key and cls.match_foreign_key(desc, col, table):
                            if relation_map.default_graph_link:
                                set_relation('graph_default')
                            else:
                                set_relation('tc_foreign_key')
                        if relation_map.tc_table_match:
                            col_table = desc['column_to_table'][str(col)]
                            if col_table == table:
                                if relation_map.default_graph_link:
                                    set_relation('graph_default')
                                elif col in desc['primary_keys']:
                                    set_relation('tc_primary_key')
                                else:
                                    set_relation('tc_table_match')
                            elif col_table is None:
                                raise RuntimeError
                                set_relation('tc_any_table')

                    elif j_type[0] == 'table':
                        table1, table2 = i_type[1], j_type[1]
                        if table1 == table2:
                            set_relation(('cc_dist', clamp(j - i, relation_map.cc_max_dist)))
                        else:
                            set_relation('cc_default')
                            if relation_map.tt_foreign_key:
                                forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                                backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                                if (forward or backward) and relation_map.default_graph_link:
                                    set_relation('graph_default')
                                elif forward and backward:
                                    set_relation('tt_foreign_key_both')
                                elif forward:
                                    set_relation('tt_foreign_key_forward')
                                elif backward:
                                    set_relation('tt_foreign_key_backward')
                    elif j_type[0] == 'value':
                        set_relation('cc_default')

                elif j_type[0] == 'column':
                    table, col = i_type[1], j_type[1]
                    set_relation('tc_default')

                    if relation_map.tc_foreign_key and cls.match_foreign_key(desc, col, table):
                        set_relation('tc_foreign_key')
                    if relation_map.tc_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('tc_primary_key')
                            else:
                                set_relation('tc_table_match')
                        elif col_table is None:
                            set_relation('tc_any_table')
                elif j_type[0] == 'table':
                    table1, table2 = i_type[1], j_type[1]
                    if table1 == table2:
                        set_relation(('tt_dist', clamp(j - i, relation_map.tt_max_dist)))
                    else:
                        set_relation('tt_default')
                        if relation_map.tt_foreign_key:
                            forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                            backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                            if forward and backward:
                                set_relation('tt_foreign_key_both')
                            elif forward:
                                set_relation('tt_foreign_key_forward')
                            elif backward:
                                set_relation('tt_foreign_key_backward')
                elif j_type[0] == 'value':
                    set_relation('tv_default')

            elif i_type[0] == 'value':
                i_real = i - v_base
                assert i_real >= 0 , (i_real)
                if j_type[0] == 'question':
                    if f"{j},{i_real}" in sc_link["q_val_match"]:
                        set_relation("vq" + sc_link["q_val_match"][f"{j},{i_real}"])
                    elif relation_map.default_link_and_graph:
                        set_relation('cq_default')
                        if relation_map.merge_sc_link:
                            if f"{j},{i_real}" in sc_link["q_col_match"]:
                                set_relation("cq" + sc_link["q_col_match"][f"{j},{i_real}"])
                    else:
                        set_relation('vq_default')
                elif j_type[0] == 'column':
                    j_real = j - c_base
                    assert i_real >= 0 and j_real >= 0, (i_real, j_real)
                    if relation_map.default_link_and_graph:
                        set_relation('cc_default')
                        assert i_real >= 0 and j_real >= 0, (i_real, j_real)
                        if f"{j_real},{i_real}" in sc_link["col_val_match"]:
                            set_relation("vc" + sc_link["col_val_match"][f"{j_real},{i_real}"])
                    elif f"{j_real},{i_real}" in sc_link["col_val_match"]:
                        set_relation("vc" + sc_link["col_val_match"][f"{j_real},{i_real}"])
                    else:
                        set_relation('vc_default')
                elif j_type[0] == 'table':
                    if relation_map.default_link_and_graph:
                        set_relation('cc_default')
                    else:
                        set_relation('vt_default')
                elif j_type[0] == 'value':
                    val1, val2 = i_type[1], j_type[1]
                    if val1 == val2:
                        if relation_map.default_link_and_graph:
                            set_relation(('cc_dist', clamp(j - i, relation_map.cc_max_dist)))
                        else:
                            set_relation(('vv_dist', clamp(j - i, relation_map.vv_max_dist)))
                    else:
                        if relation_map.default_link_and_graph:
                            set_relation('cc_default')
                        else:
                            set_relation('vv_default')
        return relations

    @classmethod
    def match_foreign_key(cls, desc, col, table):
        foreign_key_for = desc['foreign_keys'].get(str(col))
        if foreign_key_for is None:
            return False

        foreign_table = desc['column_to_table'][str(foreign_key_for)]
        return desc['column_to_table'][str(col)] == foreign_table


class NoOpUpdate:
    def __init__(self, device, hidden_size):
        pass

    def __call__(self, desc, q_enc, c_enc, c_boundaries, t_enc=None, t_boundaries=None):
        # return q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc.transpose(0, 1)
        return q_enc, c_enc, t_enc

    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc=None, t_boundaries=None):
        """
        The same interface with RAT
        return: encodings with size: length * embed_size, alignment matrix
        """
        t_enc_t = t_enc.transpose(0, 1) if t_enc is not None else t_enc
        return q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc_t, (None, None)

    def forward_batched(self, relations_batch, q_enc_batch, grnd_enc_batch):
        return  q_enc_batch, grnd_enc_batch, grnd_enc_batch, None