import torch
from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer.transformer_legacy import base_architecture
from collections import namedtuple
import sacrebleu
from fairseq import search, utils
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.ema_module import EMAModule, EMAModuleConfig
import fairseq.sequence_generator as sc
from torch.nn.utils.rnn import pad_sequence


@register_model("reranker")
class Reranker(TransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        try:
            self.decoder.load_state_dict({key.replace('decoder.', ''): val for key, val in torch.load(cfg.teacher_path)['model'].items() if key.startswith('decoder')})
            self.encoder.load_state_dict({key.replace('encoder.', ''): val for key, val in torch.load(cfg.teacher_path)['model'].items() if key.startswith('encoder')})
        except:
            pass
        self.cfg=cfg
        DummyTask = namedtuple('DummyTask', 'source_dictionary target_dictionary')
        dummy_task = DummyTask(source_dictionary=encoder.dictionary, target_dictionary=decoder.dictionary)
        self.teacher = [TransformerModel.build_model(cfg, dummy_task).cuda().half()]
        self.teacher[0].load_state_dict(torch.load(cfg.teacher_path)['model'])
        #teacher_ema_config=EMAModuleConfig()
        #teacher_ema_config.ema_decay= cfg.teacher_ema_decay
        #teacher_ema_config.ema_fp32=True
        #self.teacher_ema=EMAModule(self.teacher[0], teacher_ema_config, device='cuda')
        self.generator = [sc.SequenceGenerator(
            self.teacher,
            self.decoder.dictionary,
            beam_size=getattr(cfg, "teacher_beam", 5),
            max_len_a=getattr(cfg, "teacher_max_len_a", 1.2),
            max_len_b=getattr(cfg, "teacher_max_len_b", 10),
            min_len=getattr(cfg, "min_len", 1),
            normalize_scores=(not getattr(cfg, "unnormalized", False)),
            len_penalty=getattr(cfg, "lenpen", 1),
            unk_penalty=getattr(cfg, "unkpen", 0),
            temperature=getattr(cfg, "temperature", 1.0),
            match_source_len=getattr(cfg, "match_source_len", False),
            no_repeat_ngram_size=getattr(cfg, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(decoder.dictionary),
        )]
        
    @staticmethod
    def add_args(parser):
        super(Reranker, Reranker).add_args(parser)
        parser.add_argument('--teacher-path', type=str, metavar='STR',
            help='path to the trained teacher for distillation.')
        parser.add_argument('--teacher-ema-decay', default=.9997, type=float,
            help='Teacher ema decay.')
        parser.add_argument('--teacher-max-len-a', default=1.2, type=float,
            help='Teacher length scale from src to ref.')
        parser.add_argument('--teacher-max-len-b', default=10, type=int,
            help='Teacher length bias from src to ref.')
        parser.add_argument('--teacher-beam', default=5, type=int,
            help='Number of samples to be reranked by teacher.')
        parser.add_argument('--valid-per-epoch', default=0, type=int,
            help='Number of validation done per epoch.')


    def bleu_rerank(self, gen_toks, tgt_tokens):
        for i in range(len(gen_toks)):
            target_str = self.decoder.dictionary.string(tgt_tokens[i])
            for h in gen_toks[i]:
                h_str = self.decoder.dictionary.string(h['tokens']) 
                h['score']= sacrebleu.sentence_bleu(h_str, [target_str], tokenize="none").score
            gen_toks[i] = sorted(gen_toks[i], key=(lambda y: y['score']), reverse=True)
        return pad_sequence([g[0]['tokens'] for g in gen_toks], padding_value=self.decoder.dictionary.pad(), batch_first=True)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_tokens,
        return_all_hiddens: bool = True,
    ):
        #if self.training:
        #    self.teacher_ema.step(self)
        #    self.teacher[0]=self.teacher_ema.reverse(self.teacher[0])
        self.teacher[0].eval()   
        gen_toks = tgt_tokens     
        if self.training:
            with torch.no_grad():
                sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}, "target": tgt_tokens}
                gen_toks = self.generator[0].generate(self.teacher, sample)
                gen_toks = self.bleu_rerank(gen_toks, tgt_tokens)
                prev_output_tokens = gen_toks.clone()
                prev_output_tokens[:,1:] = gen_toks[:,:-1]
                prev_output_tokens[prev_output_tokens==self.decoder.dictionary.eos()] = self.decoder.dictionary.pad()
                prev_output_tokens[:,0] = self.decoder.dictionary.eos()
        #self.eval()
        return super().forward(src_tokens, src_lengths, prev_output_tokens, return_all_hiddens), gen_toks, prev_output_tokens

@register_model_architecture("reranker", "reranker")
def reranker_base_architecture(args):
    base_architecture(args)
