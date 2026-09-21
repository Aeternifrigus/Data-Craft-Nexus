import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV, detectDelimiter, decodeBytes, normalizeDecimal, cleanHeader, MAX_ROWS } from '../site/js/csv.js';
import { readFixture } from './helpers.js';

test('reads a plain comma-separated file', () => {
  const { head, body, delimiter } = parseCSV('a,b,c\n1,2,3\n4,5,6\n');
  assert.equal(delimiter, ',');
  assert.deepEqual(head, ['a', 'b', 'c']);
  assert.deepEqual(body, [['1', '2', '3'], ['4', '5', '6']]);
});

test('detects semicolon, tab and pipe delimiters', () => {
  assert.equal(detectDelimiter('a;b;c\n1;2;3\n'), ';');
  assert.equal(detectDelimiter('a\tb\n1\t2\n'), '\t');
  assert.equal(detectDelimiter('a|b|c\n1|2|3\n'), '|');
});

test('a semicolon file with decimal commas is not split on the commas', () => {
  const { head, body, delimiter, decimalComma } = parseCSV(readFixture('semicolon.csv'));
  assert.equal(delimiter, ';');
  assert.deepEqual(head, ['kwota', 'miasto', 'status']);
  assert.ok(body.every(r => r.length === 3));
  assert.ok(decimalComma);
  assert.ok(body.every(r => !Number.isNaN(Number(r[0]))), 'amounts should read as numbers');
});

test('a single-column file falls back to comma', () => {
  assert.equal(detectDelimiter('value\n1\n2\n'), ',');
  assert.deepEqual(parseCSV('value\n1\n2\n').body, [['1'], ['2']]);
});

test('quoted fields keep delimiters, escaped quotes and line breaks', () => {
  const { body } = parseCSV('name,comment\nx,"Said ""hi"", then left"\ny,"line one\nline two"\n');
  assert.deepEqual(body, [['x', 'Said "hi", then left'], ['y', 'line one\nline two']]);
});

test('the quoted fixture parses into three columns on every row', () => {
  const { head, body } = parseCSV(readFixture('quoted.csv'));
  assert.equal(head.length, 3);
  assert.ok(body.every(r => r.length === 3));
  assert.equal(body[0][1], 'Said "hi", then left, twice');
});

test('CRLF line endings, a byte order mark and blank lines are handled', () => {
  const { head, body } = parseCSV('﻿a,b\r\n1,2\r\n\r\n3,4\r\n');
  assert.deepEqual(head, ['a', 'b']);
  assert.deepEqual(body, [['1', '2'], ['3', '4']]);
});

test('unquoted fields are trimmed, quoted ones are kept as written', () => {
  assert.deepEqual(parseCSV('a,b\n  1 ," 2 "\n').body, [['1', ' 2 ']]);
});

test('blank and repeated headers get unique names', () => {
  assert.deepEqual(cleanHeader(['id', '', 'id', 'id', 'id_2']), ['id', 'column_2', 'id_2', 'id_3', 'id_2_2']);
});

test('decimal commas are converted only when they are whole numbers', () => {
  assert.equal(normalizeDecimal('12,50'), '12.50');
  assert.equal(normalizeDecimal('-0,5'), '-0.5');
  assert.equal(normalizeDecimal('1.234,56'), '1234.56');
  assert.equal(normalizeDecimal('1.234.567,8'), '1234567.8');
  assert.equal(normalizeDecimal('Gdynia, PL'), 'Gdynia, PL');
  assert.equal(normalizeDecimal('12,5 kg'), '12,5 kg');
  assert.equal(normalizeDecimal('1,2,3'), '1,2,3');
});

test('comma-separated files are never decimal-converted', () => {
  const { body, decimalComma } = parseCSV('a,b\n"12,50",x\n');
  assert.deepEqual(body, [['12,50', 'x']]);
  assert.equal(decimalComma, false);
});

test('reads at most MAX_ROWS data rows and says so', () => {
  const text = 'a\n' + Array.from({ length: MAX_ROWS + 10 }, (_, i) => i).join('\n');
  const { body, truncated } = parseCSV(text);
  assert.equal(body.length, MAX_ROWS);
  assert.equal(truncated, true);
  assert.equal(parseCSV('a\n1\n2\n').truncated, false);
});

test('an empty file gives no header and no rows', () => {
  assert.deepEqual(parseCSV('').head, []);
  assert.deepEqual(parseCSV('\n\n').body, []);
});

test('UTF-8 is decoded as UTF-8, with the byte order mark removed', () => {
  const bytes = new Uint8Array([0xEF, 0xBB, 0xBF, ...new TextEncoder().encode('miasto\nŁódź\n')]);
  assert.deepEqual(decodeBytes(bytes), { text: 'miasto\nŁódź\n', encoding: 'UTF-8' });
});

test('Windows-1250 files (Polish Excel exports) are decoded correctly', () => {
  // "Łódź" in Windows-1250: Ł=0xA3, ó=0xF3, d, ź=0x9F
  const bytes = new Uint8Array([0x6D, 0x69, 0x61, 0x73, 0x74, 0x6F, 0x0A, 0xA3, 0xF3, 0x64, 0x9F, 0x0A]);
  const { text, encoding } = decodeBytes(bytes);
  assert.equal(encoding, 'windows-1250');
  assert.equal(text, 'miasto\nŁódź\n');
});
