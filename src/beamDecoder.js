/**
 * WordBeamSearchDecoder for Azerbaijani Sign Language (AzSL).
 * Supports Soft Lexicon decoding with open-vocabulary beam expansion.
 */

export const AZ_ALPHABET = [
  'A', 'B', 'C', 'Ç', 'D', 'E', 'Ə', 'F', 'G', 'Ğ', 'H', 'X', 'I', 'İ',
  'J', 'K', 'Q', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U',
  'Ü', 'V', 'Y', 'Z'
];

export const AZ_LEXICON = [
  'MƏN', 'SƏN', 'O', 'BİZ', 'SİZ', 'ONLAR', 'NƏ', 'KİM', 'HARADA', 'NECƏ', 'NİYƏ', 'HANSI',
  'SALAM', 'SAĞOL', 'XOŞ', 'BƏLİ', 'YOX', 'BUYUR', 'TƏŞƏKKÜR',
  'BİR', 'İKİ', 'ÜÇ', 'DÖRD', 'BEŞ', 'ALTI', 'YEDDİ', 'SƏKKİZ', 'DOQQUZ', 'ON',
  'ANA', 'ATA', 'BACI', 'QARDAŞ', 'UŞAQ', 'DOST',
  'EV', 'BAKI', 'GƏNCƏ', 'ŞƏKİ', 'ŞUŞA', 'NAXÇIVAN', 'ŞƏHƏR', 'ÖLKƏ', 'MƏKTƏB', 'MEŞƏ', 'DƏNİZ', 'DAĞ',
  'SU', 'ÇÖRƏK', 'KİTAB', 'QAPI', 'MASA', 'STUL', 'MAŞIN', 'PUL', 'YOL', 'GÜL', 'AĞAC', 'QUŞ',
  'İT', 'PİŞİK', 'BALIQ', 'ÇAY', 'SÜD', 'ƏT', 'ALMA', 'ARMUD', 'ÜZÜM',
  'GÜN', 'GECƏ', 'İL', 'AY', 'HƏFTƏ', 'VAXT', 'SƏHƏR', 'AXŞAM',
  'YAXŞI', 'PİS', 'BÖYÜK', 'KİÇİK', 'GÖZƏL', 'İSTİ', 'SOYUQ', 'YENİ', 'ÇOX', 'AZ',
  'GETMƏK', 'GƏLMƏK', 'GÖRMƏK', 'BİLMƏK', 'SEVMƏK', 'İSTƏMƏK', 'YAZMAQ', 'OXUMAQ', 'YEMƏK', 'İÇMƏK', 'İŞLƏMƏK', 'DANIŞMAQ'
];

export class LexiconTrie {
  constructor(words) {
    this.root = {};
    this.words = new Set();
    if (words) {
      for (let i = 0; i < words.length; i++) this.insert(words[i]);
    }
  }

  insert(word) {
    const w = word.trim().toUpperCase();
    if (!w) return;
    this.words.add(w);
    let curr = this.root;
    for (let i = 0; i < w.length; i++) {
      const ch = w[i];
      if (!curr[ch]) curr[ch] = {};
      curr = curr[ch];
    }
    curr._isWord = true;
  }

  isValidPrefix(prefix) {
    const p = prefix.trim().toUpperCase();
    let curr = this.root;
    for (let i = 0; i < p.length; i++) {
      const ch = p[i];
      if (!curr[ch]) return false;
      curr = curr[ch];
    }
    return true;
  }

  isValidWord(word) {
    return this.words.has(word.trim().toUpperCase());
  }
}

export class WordBeamSearchDecoder {
  constructor(options = {}) {
    this.options = Object.assign(
      {
        beamWidth: 5,
        lmWeight: 0.6,
        lexiconWordBonus: 3.0,
        lexiconMode: 'soft', // 'none', 'hard', 'soft'
        words: AZ_LEXICON,
      },
      options
    );
    this.trie = new LexiconTrie(this.options.words);
    this.reset();
  }

  reset() {
    this.stepObservations = [];
    this.currentBeam = [{ sequence: '', score: 0.0, acScore: 0.0, lmScore: 0.0 }];
  }

  addStep(probDict, bigramProbFn) {
    this.stepObservations.push(probDict);
    const B = this.options.beamWidth;
    const mode = this.options.lexiconMode;
    const candidates = [];

    const tokens = Object.keys(probDict).map((k) => ({ label: k, prob: probDict[k] }));
    tokens.sort((a, b) => b.prob - a.prob);
    const topTokens = tokens.slice(0, 12);

    for (let b = 0; b < this.currentBeam.length; b++) {
      const hyp = this.currentBeam[b];
      const prevChar = hyp.sequence.length > 0 ? hyp.sequence[hyp.sequence.length - 1] : null;

      for (let t = 0; t < topTokens.length; t++) {
        const item = topTokens[t];
        if (item.prob <= 1e-12) continue;

        const newSeq = hyp.sequence + item.label;
        const newAcScore = hyp.acScore + Math.log(item.prob);

        const pLm = bigramProbFn ? bigramProbFn(prevChar, item.label) : 1.0 / AZ_ALPHABET.length;
        const newLmScore = hyp.lmScore + Math.log(Math.max(pLm, 1e-12));

        // In hard mode, prune invalid prefixes; in soft/none mode, never prune!
        if (mode === 'hard' && !this.trie.isValidPrefix(newSeq)) {
          continue;
        }

        const score = newAcScore + this.options.lmWeight * newLmScore;
        candidates.push({
          sequence: newSeq,
          score: score,
          acScore: newAcScore,
          lmScore: newLmScore,
        });
      }
    }

    if (candidates.length === 0) {
      const bestToken = tokens[0].label;
      for (let b = 0; b < this.currentBeam.length; b++) {
        const hyp = this.currentBeam[b];
        const p = Math.max(tokens[0].prob, 1e-12);
        candidates.push({
          sequence: hyp.sequence + bestToken,
          score: hyp.score + Math.log(p),
          acScore: hyp.acScore + Math.log(p),
          lmScore: hyp.lmScore,
        });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    this.currentBeam = candidates.slice(0, B);
    return this.getHypotheses();
  }

  getHypotheses() {
    const bonus = this.options.lexiconWordBonus;
    const mode = this.options.lexiconMode;

    return this.currentBeam
      .map((hyp) => {
        const isWord = this.trie.isValidWord(hyp.sequence);
        const isPre = this.trie.isValidPrefix(hyp.sequence);
        const wordBonus = (mode === 'soft' || mode === 'hard') && isWord ? bonus : 0.0;
        return {
          sequence: hyp.sequence,
          score: hyp.score + wordBonus,
          isWord: isWord,
          isPrefix: isPre,
        };
      })
      .sort((a, b) => b.score - a.score);
  }

  getTopWord() {
    const hyps = this.getHypotheses();
    return hyps.length > 0 ? hyps[0].sequence : '';
  }
}

