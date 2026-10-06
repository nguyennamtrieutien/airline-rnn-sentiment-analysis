'use strict';
const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const paths = {
 users:['M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2','M13 3a4 4 0 0 1 0 8','M22 21v-2a4 4 0 0 0-3-3.87','M13 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0'],
 user:['M20 21v-2a7 7 0 0 0-14 0v2','M17 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0'],
 report:['M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z','M14 2v6h6','M8 13h8M8 17h6'],
 book:['M4 4h6a2 2 0 0 1 2 2v15a3 3 0 0 0-3-3H4z','M20 4h-6a2 2 0 0 0-2 2v15a3 3 0 0 1 3-3h5z'],
 tag:['M20 13l-7 7a2 2 0 0 1-3 0L3 13V4h9l8 6a2 2 0 0 1 0 3z','M7 8h.01'],
 chart:['M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z','M7 17V9M12 17V6M17 17v-5'],
 layers:['M12 3L2 8l10 5 10-5z','M2 12l10 5 10-5M2 17l10 5 10-5'],
 settings:['M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1z','M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0'],
 database:['M20 5c0 2-4 3-8 3S4 7 4 5s4-3 8-3 8 1 8 3z','M4 5v14c0 2 4 3 8 3s8-1 8-3V5','M4 12c0 2 4 3 8 3s8-1 8-3'],
 box:['M12 3l9 5v9l-9 5-9-5V8z','M3 8l9 5 9-5M12 13v9M7 5l9 5'],
 edit:['M16 3l5 5-13 13H3v-5z','M13 6l5 5'],
 info:['M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0','M12 11v6M12 7h.01'],
 send:['M22 2L9 15M22 2l-8 20-5-7-7-5z'],
 hash:['M4 8h16M3 16h16M10 3L6 21M18 3l-4 18'],
 clock:['M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0','M12 6v6l4 2'],
 code:['M8 4L2 12l6 8M16 4l6 8-6 8'],
 smile:['M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0','M8 8h.01M16 8h.01M7 14q5 7 10 0'],
 sad:['M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0','M8 8h.01M16 8h.01M7 17q5-7 10 0'],
 neutral:['M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0','M8 8h.01M16 8h.01M8 16h8'],
 check:['M20 6L9 17l-5-5']
};
function icon(name){const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');for(const d of paths[name]||paths.info){const p=document.createElementNS(ns,'path');p.setAttribute('d',d);svg.append(p);}return svg;}
document.querySelectorAll('[data-icon]').forEach(node=>node.append(icon(node.dataset.icon)));
const colors = {negative:'#f15c6c',neutral:'#438dfa',positive:'#59a98e'};
const labels = {negative:'Negative',neutral:'Neutral',positive:'Positive'};
const feelings = {negative:'Văn bản thể hiện cảm xúc tiêu cực.',neutral:'Văn bản thể hiện cảm xúc trung tính.',positive:'Văn bản thể hiện cảm xúc tích cực.'};
const titles = ['Văn bản đầu vào','Tiền xử lý văn bản','Token hóa','Chuyển token thành chỉ số','Padding / Vector hóa','Dự đoán bằng mô hình RNN','Softmax đầu ra','Kết luận cuối cùng'];
const descriptions = ['Câu phản hồi gốc từ người dùng.','Làm sạch P0, chuyển chữ thường và giữ phủ định.','Tách văn bản thành các token (từ).','Ánh xạ từng token thành ID trong vocabulary.','Đưa về độ dài cố định; PAD=0 được mask.','Forward pass với trọng số đã huấn luyện.','Hiển thị xác suất từ cùng forward pass.','Chọn lớp có xác suất cao nhất bằng argmax.'];
let generation=0, controller, modelInfo, ready=false;
const backend=window.AirlineDemoBackend||{
 async project(){const response=await fetch('/api/project');if(!response.ok)throw Error('Không tải được thông tin đề tài.');return response.json();},
 async health(){const response=await fetch('/api/health');if(!response.ok)throw Error('Không kết nối được mô hình.');return response.json();},
 async *analyze(text,signal){
  const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal});
  if(!response.ok){const failure=await response.json();throw Error(failure.error||'Không thể phân tích.');}
  const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
  try{while(true){const chunk=await reader.read();pending+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});const lines=pending.split('\n');pending=lines.pop();for(const line of lines)if(line.trim())yield JSON.parse(line);if(chunk.done)break;}if(pending.trim())yield JSON.parse(pending);}
  finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
 }
};
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function value(body,text){body.append(el('p',text,'value-box'));}
function detail(body,title,data){const d=el('details');d.append(el('summary',title),el('pre',typeof data==='string'?data:JSON.stringify(data,null,2)));body.append(d);}
function resetSteps(){$('steps').replaceChildren(...titles.map((title,i)=>{const li=el('li',undefined,'step');li.id='step-'+(i+1);const row=el('div',undefined,'step-row'),head=el('div',undefined,'step-head');head.append(el('strong',title),el('p',descriptions[i],'step-description'),el('span','Chờ xử lý','step-state'));const content=el('div',undefined,'step-content');content.append(el('p','Chưa có dữ liệu của lượt phân tích.','waiting'));row.append(head,content);li.append(el('span',String(i+1),'step-number'),row);return li;}));$('progress').value=0;$('progress-label').textContent='0 / 8 bước';}
function resetResult(){$('output').hidden=true;$('result-placeholder').hidden=false;$('result-placeholder').textContent='Nhấn “Phân tích cảm xúc” để xem kết quả của câu bạn nhập.';for(const id of ['inference-tokens','inference-length','inference-sentiment','inference-score','inference-time','stat-time'])$(id).textContent='—';$('inference-text').textContent='Chưa phân tích';$('inference-sentiment').style.color='';}
function scoreRows(parent,probabilities){parent.replaceChildren();for(const [name,score] of Object.entries(probabilities)){const row=el('div',undefined,'score-row'),track=el('div',undefined,'track'),fill=el('div',undefined,'fill');fill.style.width=(score*100)+'%';fill.style.background=colors[name];track.append(fill);row.append(el('span',labels[name]),track,el('span',(score*100).toFixed(1)+'%','score-value'));parent.append(row);}}
function renderStep(step,data){const li=$('step-'+step),body=li.querySelector('.step-content'),desc=li.querySelector('.step-description');body.replaceChildren();
 if(step===1){desc.textContent=data.characters+' ký tự · Đầu vào hợp lệ';value(body,data.original_text);}
 if(step===2){value(body,data.clean_text);if(data.empty_marker)body.append(el('p','Không còn từ sau cleaning, dùng emptymarker.','warning'));}
 if(step===3){value(body,'[ '+data.words.join(', ')+' ]');desc.textContent=data.words.length+' token, theo tokenizer của checkpoint.';}
 if(step===4){value(body,'[ '+data.ids_before_padding.join(', ')+' ]');desc.textContent=data.token_length+' token · '+data.unknown_tokens+' OOV · vocabulary '+data.vocabulary_size.toLocaleString('en-US');detail(body,'Xem ánh xạ token, ID và OOV',data.tokens);}
 if(step===5){const preview=data.ids.length>12?data.ids.slice(0,12).join(', ')+', …':data.ids.join(', ');value(body,'[ '+preview+' ]  (tổng '+data.sequence_length+' phần tử)');desc.textContent=data.retained_tokens+' token + '+data.padding_tokens+' PAD; '+data.padding+'-padding.';if(data.removed_tokens)body.append(el('p','Cắt '+data.removed_tokens+' token ở '+(data.truncating==='post'?'cuối':'đầu')+' chuỗi.','warning'));detail(body,'Xem chuỗi đầy đủ và mask',{token_ids:data.ids,mask:data.mask,shape:data.shape,quy_uoc:'true = token thật, false = PAD'});}
 if(step===6){const flow=el('div',undefined,'model-flow');const nodes=[['Chuỗi số','1 × '+data.embedding_shape[1]],['Embedding',data.embedding_shape[1]+' × '+data.embedding_dimension],['RNN',data.hidden_units+' units'],['Dense',data.dense_shape[1]+' units'],['Softmax',data.output_shape[1]+' lớp']];nodes.forEach(([name,size],i)=>{if(i)flow.append(el('span','→','flow-arrow'));const item=el('div',undefined,'model-node');item.append(el('strong',name),el('span',size));flow.append(item);});body.append(flow);desc.textContent=data.parameters.toLocaleString('en-US')+' tham số · tanh · inference, dropout tắt.';detail(body,'Xem giá trị vector thật',{embedding_8_chieu_dau:data.first_token_embedding_preview,RNN_8_chieu_dau:data.last_hidden_state_preview,hidden_L2:data.hidden_l2_norm,activation:data.activation,dense_activation:data.dense_activation,note:data.note});}
 if(step===7){const scores=el('div',undefined,'mini-scores');scoreRows(scores,data.probabilities);body.append(scores);desc.textContent='Ba lớp cảm xúc · Tổng xác suất ≈ '+data.probability_sum.toFixed(4)+'.';}
 if(step===8){const conclusion=el('div',undefined,'conclusion '+data.sentiment);conclusion.append(icon('check'));const content=el('div');content.append(el('strong',labels[data.sentiment]),el('small',feelings[data.sentiment]));conclusion.append(content,el('span',(data.confidence*100).toFixed(1)+'%','conclusion-score'));body.append(conclusion);desc.textContent='Argmax → '+labels[data.sentiment]+'.';}
}
function seconds(ms){return (ms/1000).toLocaleString('vi-VN',{minimumFractionDigits:3,maximumFractionDigits:3})+' giây';}
function showResult(data,elapsed){$('sentiment').textContent=labels[data.sentiment];$('confidence').textContent=(data.confidence*100).toFixed(1)+'%';$('prediction-banner').className='prediction-banner '+data.sentiment;$('sentiment-symbol').replaceChildren(icon({positive:'smile',negative:'sad',neutral:'neutral'}[data.sentiment]));scoreRows($('scores'),data.probabilities);$('checkpoint').textContent=JSON.stringify({config:data.config_id,seed:data.seed,clean_text:data.clean_text,token_length:data.token_length,unknown_tokens:data.unknown_tokens,truncated:data.truncated},null,2);$('inference-text').textContent=data.original_text;$('inference-tokens').textContent=data.token_length;$('inference-length').textContent=data.sequence_length;$('inference-sentiment').textContent=labels[data.sentiment];$('inference-sentiment').style.color={positive:'#078858',negative:'#b63750',neutral:'#3672c2'}[data.sentiment];$('inference-score').textContent=(data.confidence*100).toFixed(1)+'%';$('inference-time').textContent=seconds(elapsed);$('stat-time').textContent=seconds(elapsed);$('output').hidden=false;$('result-placeholder').hidden=true;}
function accept(event){if(event.type==='step_start'){const li=$('step-'+event.step);li.className='step active';li.querySelector('.step-state').textContent='Đang xử lý';$('run-status').textContent='Đang thực hiện bước '+event.step+': '+titles[event.step-1];}else if(event.type==='step_done'){const li=$('step-'+event.step);renderStep(event.step,event.data);li.className='step complete';li.querySelector('.step-state').textContent='Hoàn tất · '+event.elapsed_ms.toFixed(2)+' ms';$('progress').value=event.step;$('progress-label').textContent=event.step+' / 8 bước';}else if(event.type==='result'){showResult(event.result,event.total_elapsed_ms);$('run-status').textContent='Đã hoàn thành 8 khâu phân tích cho câu vừa nhập.';}else if(event.type==='error'){throw Error(event.error);}}
function busy(value){$('submit').disabled=value||!ready;$('submit-label').textContent=value?'Đang phân tích…':'Phân tích cảm xúc';document.querySelectorAll('.sample').forEach(x=>x.disabled=value);}
$('tweet').addEventListener('input',()=>{$('character-count').textContent=$('tweet').value.length.toLocaleString('vi-VN')+' / 4.000 ký tự';});
document.querySelectorAll('.sample').forEach(button=>button.onclick=()=>{$('tweet').value=button.dataset.text;$('tweet').dispatchEvent(new Event('input'));$('tweet').focus();});
$('form').addEventListener('submit',async event=>{event.preventDefault();if(!ready)return;const current=++generation;controller?.abort();resetSteps();resetResult();$('error').hidden=true;const text=$('tweet').value;
 if(!text.trim()){$('step-1').className='step failed';$('step-1').querySelector('.step-state').textContent='Đầu vào không hợp lệ';$('run-status').textContent='Chưa chạy mô hình. Hãy nhập văn bản hợp lệ.';$('error').textContent='Hãy nhập một câu hoặc tweet tiếng Anh.';$('error').hidden=false;busy(false);return;}
 controller=new AbortController();busy(true);$('result-placeholder').textContent='Đang xử lý phản hồi…';$('run-status').textContent='Đang phân tích câu với mô hình RNN…';let receivedResult=false;
 try{for await(const item of backend.analyze(text,controller.signal)){if(current!==generation)return;accept(item);if(item.type==='result')receivedResult=true;if(window.AirlineDemoBackend&&item.type==='step_start')await new Promise(requestAnimationFrame);}if(!receivedResult)throw Error('Kết nối dừng trước khi hoàn thành phân tích.');}
 catch(error){if(current!==generation||error.name==='AbortError')return;$('error').textContent=error.message;$('error').hidden=false;$('result-placeholder').textContent='Chưa có kết quả. Kiểm tra thông báo lỗi và thử lại.';$('run-status').textContent='Phân tích chưa hoàn tất. Kiểm tra thông báo lỗi và thử lại.';document.querySelectorAll('.step.active').forEach(li=>{li.className='step failed';li.querySelector('.step-state').textContent='Chưa hoàn tất';});}
 finally{if(current===generation)busy(false);}
});
resetSteps();resetResult();$('tweet').dispatchEvent(new Event('input'));
Promise.all([backend.project(),backend.health()]).then(([info,health])=>{if(health.status!=='ready')throw Error('Mô hình chưa sẵn sàng.');modelInfo=health;ready=true;$('school').textContent=info.school;$('footer-school').textContent=info.school;$('report-title').textContent=info.title;$('teacher').textContent=info.teacher;$('class-name').textContent=info.class_name;$('footer-year').textContent=info.course+' · RNN · '+info.year;$('members').replaceChildren(...info.members.map(m=>{const li=el('li');li.append(icon('user'),el('span',m.student_id+' - '+m.name));return li;}));$('health-status').querySelector('span').textContent='Demo sẵn sàng';for(const key of ['vocabulary_size','sequence_length'])document.querySelectorAll('[data-value="'+key+'"]').forEach(n=>n.textContent=health[key].toLocaleString('en-US'));$('report-classes').textContent=$('stat-classes').textContent=health.label_names.length;busy(false);}).catch(error=>{$('health-status').classList.add('offline');$('health-status').querySelector('span').textContent='Chưa kết nối';$('error').textContent=error.message+' Hãy kiểm tra kết nối mạng và tải lại trang.';$('error').hidden=false;});
