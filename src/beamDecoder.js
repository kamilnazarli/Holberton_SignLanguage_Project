/**
 * Word-Level Beam Search Decoder for Azerbaijani Sign Language (AzSL).
 * Reusable JavaScript module for client-side and Node.js environments.
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
  'GETMƏK', 'GƏLMƏK', 'GÖRMƏK', 'BİLMƏK', 'SEVMƏK', 'İSTƏMƏK', 'YAZMAQ', 'OXUMAQ', 'YEMƏK', 'İÇMƏK', 'İŞLƏMƏK', 'DANIŞMAQ',
];

export class LexiconTrie {
  constructor(words = AZ_LEXICON) {
    this.root = {};
    this.words = new Set();
    for (let i = 0; i < words.length; i++) {
      this.insert(words[i]);
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

export class BigramLM {
  constructor(words = AZ_LEXICON, alpha = 0.3) {
    this.alpha = alpha;
    this.alphabetSize = AZ_ALPHABET.length;
    this.counts = {};
    this.totals = {};
    this.unigrams = {};
    this.totalUnigrams = 0;

    for (let w = 0; w < words.length; w++) {
      const word = words[w].trim().toUpperCase();
      for (let i = 0; i < word.length; i++) {
        const ch = word[i];
        this.unigrams[ch] = (this.unigrams[ch] || 0) + 1;
        this.totalUnigrams++;
        if (i > 0) {
          const prev = word[i - 1];
          this.counts[prev] = this.counts[prev] || {};
          this.counts[prev][ch] = (this.counts[prev][ch] || 0) + 1;
          this.totals[prev] = (this.totals[prev] || 0) + 1;
        }
      }
    }
  }

  logProb(curr, prev = null) {
    if (!prev || !this.totals[prev]) {
      const count = this.unigrams[curr] || 0;
      const p = (count + this.alpha) / (this.totalUnigrams + this.alpha * this.alphabetSize);
      return Math.log(Math.max(p, 1e-12));
    }
    const count = (this.counts[prev] && this.counts[prev][curr]) || 0;
    const total = this.totals[prev] || 0;
    const p = (count + this.alpha) / (total + this.alpha * this.alphabetSize);
    return Math.log(Math.max(p, 1e-12));
  }
}

export class WordBeamSearchDecoder {
  constructor(options = {}) {
    this.beamWidth = options.beamWidth || 5;
    this.lmWeight = options.lmWeight !== undefined ? options.lmWeight : 0.6;
    this.lexiconWordBonus = options.lexiconWordBonus !== undefined ? options.lexiconWordBonus : 3.0;
    this.strictLexiconPrefix = options.strictLexiconPrefix !== undefined ? options.strictLexiconPrefix : true;
    const words = options.lexiconWords || AZ_LEXICON;

    this.trie = new LexiconTrie(words);
    this.lm = new BigramLM(words, 0.3);
    this.reset();
  }

  reset() {
    this.stepObservations = [];
    this.currentBeam = [{ sequence: '', score: 0.0, acScore: 0.0, lmScore: 0.0 }];
  }

  addStep(probDict) {
    this.stepObservations.push(probDict);
    const B = this.beamWidth;
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
        const lmDelta = this.lm.logProb(item.label, prevChar);
        const newLmScore = hyp.lmScore + lmDelta;

        const isPre = this.trie.isValidPrefix(newSeq);
        if (this.strictLexiconPrefix && !isPre) {
          continue;
        }

        const score = newAcScore + (this.lmWeight * newLmScore);
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
    return this.currentBeam.map((hyp) => {
      const isWord = this.trie.isValidWord(hyp.sequence);
      const isPre = this.trie.isValidPrefix(hyp.sequence);
      const bonus = isWord ? this.lexiconWordBonus : 0;
      return {
        sequence: hyp.sequence,
        score: hyp.score + bonus,
        isWord: isWord,
        isPrefix: isPre,
      };
    }).sort((a, b) => b.score - a.score);
  }

  getTopWord() {
    const hyps = this.getHypotheses();
    return hyps.length > 0 ? hyps[0].sequence : '';
  }
}

