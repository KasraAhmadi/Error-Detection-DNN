// Simulate an Arduino UNO (ATmega328P @16 MHz) firmware with avr8js and print its serial output.
// usage: node run.mjs firmware.hex [stopToken]
import fs from 'fs';
import { CPU, avrInstruction, AVRTimer, timer0Config, timer1Config, timer2Config, AVRUSART, usart0Config } from 'avr8js';

const [hexPath, stop = '== END =='] = process.argv.slice(2);
const prog = new Uint16Array(0x8000 / 2);
const bytes = new Uint8Array(prog.buffer);
for (const line of fs.readFileSync(hexPath, 'utf8').split('\n')) {
  if (line[0] !== ':') continue;
  const n = parseInt(line.substr(1, 2), 16), addr = parseInt(line.substr(3, 4), 16), type = parseInt(line.substr(7, 2), 16);
  if (type !== 0) continue;
  for (let i = 0; i < n; i++) bytes[addr + i] = parseInt(line.substr(9 + i * 2, 2), 16);
}
const cpu = new CPU(prog);
new AVRTimer(cpu, timer0Config); new AVRTimer(cpu, timer1Config); new AVRTimer(cpu, timer2Config);
const usart = new AVRUSART(cpu, usart0Config, 16e6);
let out = '';
usart.onByteTransmit = (b) => { const c = String.fromCharCode(b); out += c; process.stdout.write(c); };
const t0 = Date.now();
while (!out.includes(stop)) {
  avrInstruction(cpu);
  cpu.tick();
  if (cpu.cycles > 16e6 * 600) { console.error('\n[timeout: 600 s simulated]'); break; }
}
console.error(`\n[simulated ${(cpu.cycles / 16e6).toFixed(2)} s of MCU time in ${((Date.now() - t0) / 1000).toFixed(0)} s]`);
