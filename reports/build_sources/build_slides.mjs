import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT=process.env.AIRLINE_PROJECT_ROOT??path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const TMP=path.join(ROOT,'.build/rnn-rebrand/slide-authoring');
await fs.mkdir(TMP,{recursive:true});
const SKILL=process.env.PRESENTATIONS_SKILL_PATH??'/Users/johnnycu/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const PY=process.env.REPORT_PYTHON??'/Users/johnnycu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES??='/Users/johnnycu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT=resolvePresentationFont({fontFamily:'Arial'});
const GROUP=JSON.parse(await fs.readFile(path.join(ROOT,'airline_rnn/web/project-info.json'),'utf8'));
const P=Presentation.create({slideSize:{width:1280,height:720}});
const N='#16344C',BLUE='#28658C',TEAL='#2E8C7E',MUTED='#526B7D',LIGHT='#EEF3F6',GOLD='#B67831';
const slides=[],texts=[],chartOwners=[],tableOwners=[];
function csv(s){const lines=s.trim().split('\n');const parse=l=>{const r=[];let v='',q=false;for(let i=0;i<l.length;i++){const c=l[i];if(c==='"'){if(q&&l[i+1]==='"'){v+='"';i++;}else q=!q;}else if(c===','&&!q){r.push(v);v='';}else v+=c;}r.push(v);return r;};const h=parse(lines[0]);return lines.slice(1).map(l=>Object.fromEntries(parse(l).map((v,i)=>[h[i],v])));}
const V=csv(await fs.readFile(path.join(ROOT,'results/tables/validation_summary.csv'),'utf8'));
const TEST=csv(await fs.readFile(path.join(ROOT,'results/tables/test_summary.csv'),'utf8'));
const H=csv(await fs.readFile(path.join(ROOT,'models/runs/B5-balanced/seed-3407/history.csv'),'utf8'));
const metrics=JSON.parse(await fs.readFile(path.join(ROOT,'results/metrics/B5-balanced_seed-3407_test.json'),'utf8'));
function txt(s,text,x,y,w,h,size=28,color=N,bold=false){const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=text;sh.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};return sh;}
function slide(title,kicker='THỰC NGHIỆM RNN'){const s=P.slides.add();s.background.fill='#FFFFFF';slides.push(s);texts.push({number:slides.length,title});txt(s,kicker,64,30,1120,25,16,MUTED,true);txt(s,title,64,75,1152,110,43,N,true);txt(s,'Deep Learning   ·   Airline Sentiment Analysis',64,676,1000,24,16,MUTED);txt(s,String(slides.length).padStart(2,'0'),1155,672,60,28,18,MUTED);return s;}
function notes(s,t,sources=''){if(chartOwners.includes(slides.length))t+=' Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.';s.speakerNotes.textFrame.setText(t+'\n\nNguồn: '+sources);texts[texts.length-1].notes=t;texts[texts.length-1].sources=sources;}
function table(s,values,x,y,w,h,widths,size=24){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({style:'solid',fill:'#CFDCE5',width:1});t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:FONT,fontSize:size,color:N},margins:{left:12,right:12,top:10,bottom:10},anchor:'center'});for(let r=0;r<values.length;r++){for(let c=0;c<values[0].length;c++){const z=t.getCell(r,c);z.fill=r===0?N:r%2?LIGHT:'#FFFFFF';z.text.style={typeface:FONT,fontSize:size,bold:r===0,color:r===0?'#FFFFFF':N};}}tableOwners.push(slides.length);return t;}
function chart(s,type,title,cats,series,x,y,w,h,{min=0,max=1,fmt='0.00',unit=.2,labels=false,legend=false}={}){const c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{typeface:FONT,fontSize:22,fill:N,bold:true},categories:cats,series:series.map(r=>({...r,values:r.values.map(v=>Number(Number(v).toFixed(6)))})),hasLegend:legend,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:19,fill:N}},barOptions:{direction:'column',grouping:'clustered',gapWidth:80},lineOptions:{smooth:false},xAxis:{textStyle:{typeface:FONT,fontSize:19,fill:N},line:{fill:'#BDCCD7',width:1}},yAxis:{min,max,majorUnit:unit,numberFormatCode:fmt,textStyle:{typeface:FONT,fontSize:18,fill:MUTED},majorGridlines:{fill:'#E3EAF0',width:1}},dataLabels:{showValue:labels,position:'outEnd',textStyle:{typeface:FONT,fontSize:21,fill:N}},chartFill:'#FFFFFF',chartLine:{fill:'none',width:0}});applyPresentationChartFont(c,{fontFamily:FONT});chartOwners.push(slides.length);return c;}
async function fig(s,name,x,y,w,h){s.images.add({blob:new Uint8Array(await fs.readFile(path.join(ROOT,'results/figures',name+'.png'))),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
// 1. Minimal academic cover, identities explicitly authorized from report template.
{
const s=P.slides.add();s.background.fill=N;slides.push(s);texts.push({number:1,title:GROUP.title});
txt(s,'TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH',64,34,1152,32,20,'#C5D9E7');
txt(s,'Xây dựng mô hình phân tích\ncảm xúc khách hàng trong\nngành hàng không sử dụng RNN',64,142,1152,240,49,'#FFFFFF',true);
txt(s,'Thực nghiệm ba lớp với quy trình tái lập',68,425,1110,55,32,'#C5D9E7');
txt(s,GROUP.members.map(m=>m.student_id+' – '+m.name).join('\n')+'\nGVHD: '+GROUP.teacher,68,506,1100,122,22,'#FFFFFF');
txt(s,'Khoa Công nghệ thông tin   ·   2026',68,645,1100,35,20,'#C5D9E7');
texts[0].title=GROUP.title;
notes(s,'Kính chào thầy và các bạn. Đề tài nghiên cứu RNN cho sentiment của tweet hàng không với ba lớp negative, neutral và positive. Trình bày tập trung vào audit nguồn, protocol tái lập, các đối chứng có kiểm soát và kết quả thực đo. Đề tài: Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN. Ba thành viên gồm Trịnh Nguyễn Anh Hào 2611307, Lê Huy Huân 2611308, Nguyễn Nam Triều Tiên 2611323.','Báo cáo Word; results/tables/test_summary.csv');}
// 2. Source audit defines the comparison before showing numbers.
{
const s=slide('Notebook nguồn thực tế dùng LSTM và hai lớp','AUDIT NGUỒN');
table(s,[['Thành phần','Notebook Kaggle','Thực nghiệm chính'],['Mạng hồi quy','LSTM196','RNN196'],['Nhãn','negative / positive','negative / neutral / positive'],['Dataset sau xử lý','6.541 tweet lọc','14.353 tweet sạch'],['Fit vocabulary','Trước split, toàn corpus','Chỉ train'],['Chọn model','Epoch cuối','Validation Macro-F1']],64,200,1152,355,[310,410,432],24);
txt(s,'90,32% là output lịch sử của nguồn. A1 là baseline chuyển thể sang RNN.',64,587,1152,64,27,BLUE,true);
notes(s,'Tên notebook ghi RNN nhưng code thực tế dùng LSTM và bỏ toàn bộ neutral, sau đó bỏ 5.000 negative đầu. Nguồn fit tokenizer trên corpus trước chia tập. Audit A1 phát hiện 68 tweet_id và 85 clean text trùng qua splits. Những khác biệt này giới hạn cách so sánh với con số 90,32%. Nhóm giữ yêu cầu RNN, vì thế gọi A1 là baseline chuyển thể, không công bố tái lập chính xác nguồn. Chưa lượng hóa mức Accuracy tăng do overlap.','https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn ; reports/source_audit.md');}
// 3. Dataset imbalance as native editable chart.
{
const s=slide('Negative chiếm 62,69% dữ liệu gốc','DATASET');
chart(s,'bar','Số tweet theo sentiment',['negative','neutral','positive'],[{name:'Tweets',values:[9178,3099,2363],fill:BLUE,valuesFormatCode:'#,##0'}],64,195,780,425,{max:10000,fmt:'#,##0',unit:2000,labels:true});
txt(s,'14.640 tweet\n15 cột dữ liệu',880,228,330,110,36,N,true);
txt(s,'Input: text\nOutput: 3 sentiments\n\nAirline dùng cho EDA\nTweet ID dùng để nhóm trùng',880,360,335,245,25,MUTED);
notes(s,'Dataset Twitter US Airline Sentiment của CrowdFlower chứa 14.640 dòng với negative 9.178, neutral 3.099 và positive 2.363. Mô hình chỉ nhận văn bản. Airline không làm feature, tweet_id hỗ trợ audit và grouping. Lớp negative chiếm đa số nên Accuracy riêng lẻ dễ che hiệu quả thấp ở neutral/positive. Majority baseline trên test sạch đạt 63,66% Accuracy nhưng Macro-F1 chỉ 0,2593.','https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment ; results/tables/dataset.csv');}
// 4. Data isolation and actual counts.
{
const s=slide('Nhóm dữ liệu trùng đi trọn vào một split','PROTOCOL');
table(s,[['Split','Negative','Neutral','Positive','Tổng'],['Train','6.404','2.095','1.544','10.043'],['Validation','1.368','458','335','2.161'],['Test','1.368','448','333','2.149']],64,195,1152,270,[300,210,210,210,222],25);
txt(s,'36 bản sao loại bỏ, 251 dòng xung đột nhãn cách ly',64,489,1152,43,28,BLUE,true);
txt(s,'Split seed 42, tỷ lệ gần 70 / 15 / 15\nVocabulary học trên train, chọn cấu hình trên validation\nTest mở sau khi đóng băng lựa chọn mô hình',64,543,1152,111,27);
notes(s,'Dữ liệu sau audit còn 14.353 tweet. Nhóm trùng được nối bắc cầu bằng tweet_id, canonical raw text hoặc clean text. Mọi thành viên nhóm cùng split nên loại được overlap theo định nghĩa này. Stratification cố gắng giữ tỷ lệ lớp, số dòng thực tế lệch nhẹ tỷ lệ lý tưởng do nhóm không thể tách. Train-only vocabulary và test gate là hai cơ chế riêng. Chúng không loại được tất cả quan hệ ngữ nghĩa hoặc hội thoại gần nhau.','results/tables/dataset_splits.csv ; results/metrics/artifact_verification.json');}
// 5. Actual network figure, editable foreground explanation.
{
const s=slide('RNN cập nhật trạng thái theo thứ tự token','MÔ HÌNH');
const layers=[['Embedding','4.000 × 128'],['Spatial','Dropout 0,5'],['RNN','196 / tanh'],['Dropout','0,2'],['Dense 100','ReLU'],['Dropout','0,4'],['Dense 3','Softmax']];
for(let i=0;i<layers.length;i++){const x=64+i*166;const sh=s.shapes.add({geometry:'rect',position:{left:x,top:218,width:152,height:139},fill:i===2?BLUE:LIGHT,line:{fill:'#A3B7C9',width:1}});txt(s,layers[i][0]+'\n'+layers[i][1],x+9,252,135,87,21,i===2?'#FFFFFF':N,true);if(i<6)txt(s,'→',x+153,269,15,36,21,MUTED);}

txt(s,'Embedding 4.000 × 128\nRNN 196 units, tanh\nDense 100 ReLU, softmax 3 lớp',64,440,720,145,31,N,true);
txt(s,'595.703',870,442,345,65,44,BLUE,true);txt(s,'tham số trainable',870,515,345,55,27,BLUE,true);
txt(s,'T=40, post-padding, mask PAD. Embedding học cùng model, không pretrained.',64,601,1152,49,26,MUTED);
notes(s,'Mỗi token tra embedding 128 chiều rồi đi qua RNN. Trạng thái hiện tại là tanh của tổng biến đổi tuyến tính embedding và trạng thái trước. Mạng lấy trạng thái cuối của chuỗi đã masking, qua Dense100 và Dense3. Riêng RNN có H(D+H+1)=63.700 tham số, embedding có 512.000 nên chiếm phần lớn tổng. Padding ID0 được mask để không đưa phần đệm vào nội dung. Toàn bộ 22 run đều là RNN.','https://keras.io/api/layers/recurrent_layers/simple_rnn/ ; https://keras.io/api/layers/core_layers/embedding/ ; results/tables/final_architecture.csv');}
// 6. Training controls.
{
const s=slide('Checkpoint chọn bằng validation Macro-F1','HUẤN LUYỆN');
table(s,[['Thành phần','Thiết lập chính'],['Seeds','42, 2026, 3407 trên cùng split'],['Optimizer / LR','Adam / 0,001, global clipnorm 1,0'],['Batch / epoch budget','32 / tối đa 20 epoch'],['Early Stopping','patience 5, min_delta 0,001'],['Dropout','Spatial 0,5, input/recurrent 0,3, sau RNN 0,2, Dense 0,4']],64,190,1152,350,[360,792],24);
txt(s,'F1 tính trên toàn validation ở chế độ inference mỗi epoch',64,570,1152,43,28,BLUE,true);
txt(s,'CPU macOS arm64, 10 cores, RAM 16 GiB, TensorFlow 2.20.0, Keras 3.11.3',64,623,1152,35,22,MUTED);
notes(s,'Checkpoint dùng Macro-F1 lớn nhất trên toàn validation, tie thì val loss thấp hơn. Early Stopping giữ một mốc riêng với min_delta, nên không nhất thiết cùng checkpoint best. History lưu metric và wall time. Dropout giúp regularization, gradient clipping giới hạn exploding gradient nhưng không chữa hoàn toàn vanishing gradient. Main model có class weighting ở B5, validation metric không nhân class weights. Tổng wall time training của 22 run khoảng 19,91 phút trên CPU này.','results/tables/final_training_configuration.json ; environment/ ; https://arxiv.org/abs/1211.5063 ; https://arxiv.org/abs/1412.6980');}
// 7. Adapted binary baselines; no direct false source comparison.
{
const s=slide('A2 cho thấy biến thiên lớn giữa training seeds','BASELINE BINARY');
table(s,[['Baseline','N test','Test Accuracy','Test Macro-F1'],['A1 chuyển thể, 1 seed','1.963','83,80%','0,8249'],['A2 sạch, 3 seeds','1.966','74,08% ± 11,12 đpt','0,6531 ± 0,2183']],64,210,1152,205,[380,170,325,277],25);
txt(s,'A2 seed 42 có F1 positive khoảng 0,0593',64,452,1152,52,32,BLUE,true);
txt(s,'Giữ đầy đủ kết quả của cả ba seeds\nA1/A2 khác dataset và protocol, chưa tách nguyên nhân\nKhông dùng test để đổi seed hoặc hyperparameter',64,525,1152,125,27);
notes(s,'A1 giữ source pipeline, chỉ chuyển mạng hồi quy sang RNN và dùng learning rate tường minh. A2 sạch hơn nhưng vẫn source cleaner và không masking/clipping; nhiều thành phần khác nên không phải ablation một yếu tố. Seed42 dừng epoch6 với checkpoint1 và thiên negative. Hai seed khác tốt hơn, vì vậy độ lệch chuẩn cao là thông tin chính. Nhóm không loại run xấu và không lấy seed 3407 làm đại diện cho mean A2.','results/tables/test_summary.csv ; results/tables/test_runs.csv ; reports/experiment_chapter.md');}
// 8. Registered main matrix.
{
const s=slide('Các đối chứng B chỉ đổi một yếu tố so với C0','EXPERIMENT MATRIX');
table(s,[['Config','Yếu tố thay đổi','Câu hỏi khảo sát'],['C0','T40, H196, không class weights','Mốc đối chứng ba lớp'],['B1-20','Sequence length 20','Cắt chuỗi tác động quality/cost thế nào?'],['B1-60','Sequence length 60','Thêm PAD có tăng chất lượng không?'],['B3-64','Hidden units 64','Dung lượng thấp có đủ không?'],['B3-128','Hidden units 128','Dung lượng trung gian có cải thiện không?'],['B5-balanced','Class weighting','Neutral/positive có được nhận tốt hơn?']],64,189,1152,394,[240,350,562],22);
txt(s,'Mỗi configuration chính chạy 3 seeds. Split, P0, vocabulary, embedding và LR giữ cố định.',64,619,1152,40,24,MUTED);
notes(s,'Ma trận chính có sáu configurations, gồm mốc C0 và năm thay đổi độc lập trong ba nhóm nghiên cứu. Tổng tính cả A1/A2 là tám configurations và 22 run. Không khảo sát dropout, learning rate, embedding dimension hoặc vocabulary size trong lần chạy này. B1-60 giữ nguyên mọi thứ trừ T, B3 thay H và B5 thay class weights. Không có test metrics cho các cấu hình B chưa thắng vì test chỉ dành cho đánh giá cuối.','configs/ ; results/tables/experiment_comparison.csv');}
// 9. Native validation chart and uncertainty note.
{
const s=slide('B5-balanced có validation Macro-F1 cao nhất','LỰA CHỌN CẤU HÌNH');
const m=V.filter(r=>r.benchmark==='three_class');
chart(s,'bar','Mean validation Macro-F1, 3 seeds',m.map(r=>r.config_id),[{name:'Macro-F1',values:m.map(r=>+r.val_macro_f1_mean),fill:BLUE,valuesFormatCode:'0.0000',points:[{idx:5,fill:TEAL}]}],64,188,1152,373,{max:.8,fmt:'0.00',unit:.2,labels:true});
txt(s,'B5: 0,6926 ± 0,0115. C0: 0,6623 ± 0,0017',64,591,1152,46,30,N,true);
txt(s,'SD phản ánh training seeds trên cùng split. Chọn config trước test.',64,643,1152,27,22,MUTED);
notes(s,'B5 đạt mean validation Macro-F1 cao nhất 0,6926. Bar biểu diễn mean, SD của từng cấu hình có trong bảng Word/CSV. B3-128 và T20 quanh 0,6724 nhưng thấp hơn B5. Rule chọn xét Macro-F1 trước, nếu gần top dưới 0,005 mới dùng tham số rồi thời gian. B5 thắng nên chưa cần tie-break. Seed demo 3407 có Macro-F1 validation trung vị, không chọn seed tốt nhất trên test.','results/tables/validation_summary.csv ; results/metrics/final_selection.json');}
// 10. Runtime chart and context limitation.
{
const s=slide('T60 tăng thời gian nhưng không thêm token thật','CHI PHÍ VÀ BIỂU DIỄN');
chart(s,'bar','Median training wall time (giây)',['T20','T40','T60','H64','H128','B5'],[{name:'Seconds',values:[39.6106,76.4869,108.8558,30.3249,50.4741,75.4644],fill:GOLD,valuesFormatCode:'0.0'}],64,195,800,392,{max:120,fmt:'0',unit:30,labels:true});
txt(s,'Max độ dài train = 34\nT40 và T60 không cắt chuỗi\nMask bỏ qua PAD',900,241,315,155,27,N,true);
txt(s,'Validation của T40/T60\ngiống nhau ở cả 3 seeds\n\nT60 tốn thêm khoảng 42%',900,423,315,155,25,MUTED);
notes(s,'Phân bố token sau P0 có median 20, max train 34. T60 không thêm ngữ cảnh so với T40 nên kết quả validation bằng nhau từng seed. Time tăng 42% là cost phần đệm ở triển khai hiện tại. H64 nhanh nhất nhưng mean Macro-F1 gần C0, H128 có chất lượng nhỉnh hơn và ít thời gian hơn H196. Chưa có benchmark latency hoặc peakRAM theo config, số trên là wall time training gồm tính metrics, không phải inference time.','results/tables/validation_summary.csv ; results/figures/F05_train_lengths.png');}
// 11. Native history chart.
{
const s=slide('Checkpoint demo ở epoch 16 trong 20 epoch','HỘI TỤ');
chart(s,'line','Macro-F1 inference theo epoch',H.map(r=>r.epoch),[{name:'Train-eval',values:H.map(r=>+r.train_eval_macro_f1),line:{fill:BLUE,width:3}},{name:'Validation',values:H.map(r=>+r.val_macro_f1),line:{fill:TEAL,width:3}}],64,190,900,410,{max:1,unit:.2,legend:true});
txt(s,'B5 seed 3407\nVal Macro-F1 best\n0,6935',1000,249,220,162,28,N,true);
txt(s,'Fit train có dropout\nF1 ở hình này tính\ninference toàn tập',1000,442,220,142,24,MUTED);
notes(s,'Đường F1 được tính ở chế độ inference toàn train/validation, không trung bình minibatch. Checkpoint 16 đạt validation Macro-F1 tốt nhất của seed 3407. Accuracy trong fit có dropout còn validation tắt, nên hai đường không cùng chế độ. B5 train loss còn có class weights, vì vậy không so trị số loss với C0 để kết luận model tốt hơn. Xem ba hình loss, Accuracy, F1 riêng trong Word cho phân tích hội tụ đầy đủ.','models/runs/B5-balanced/seed-3407/history.csv ; model_summary.txt');}
// 12. Final held-out comparison.
{
const s=slide('B5 đạt test Macro-F1 trung bình 0,6914','KẾT QUẢ CUỐI');
chart(s,'bar','Test trên cùng 2.149 tweet, mean 3 seeds',['C0','B5-balanced'],[{name:'Accuracy',values:[.7474794478,.7518225531],fill:BLUE,valuesFormatCode:'0.0000'},{name:'Macro-F1',values:[.6515387382,.6914251040],fill:TEAL,valuesFormatCode:'0.0000'}],64,193,800,398,{max:.85,unit:.2,labels:true,legend:true});
txt(s,'75,18% ± 1,34 đpt\nAccuracy',900,228,320,106,32,BLUE,true);
txt(s,'0,6914 ± 0,0158\nMacro-F1',900,368,320,106,32,TEAL,true);
txt(s,'Weighted-F1 mean 0,7568\nMajority Macro-F1 0,2593',900,520,320,81,24,MUTED);
notes(s,'B5 đạt test Accuracy trung bình 75,18%, SD 1,34 điểm phần trăm; Macro-F1 trung bình 0,6914, SD 0,0158; Weighted-F1 trung bình 0,7568. C0 đạt Macro-F1 trung bình 0,6515, nên chênh lệch khoảng 0,0399. Đây là trung bình ba training seeds trên cùng split, không phải ensemble. Ba run chưa chứng minh ý nghĩa thống kê trên những split hoặc miền khác. Checkpoint demo riêng đạt Accuracy 75,38% và Macro-F1 0,6869. Confusion matrix tiếp theo thuộc checkpoint này.','results/tables/test_summary.csv ; results/metrics/B5-balanced_seed-3407_test.json');}
// 13. Editable CM and class F1, sufficient context.
{
const s=slide('Neutral thường nhầm negative, positive nhầm neutral','PER-CLASS EVALUATION');
table(s,[['Thật / đoán','Negative','Neutral','Positive'],['Negative','1.153','170','45'],['Neutral','157','254','37'],['Positive','41','79','213']],64,215,625,288,[190,145,145,145],24);
chart(s,'bar','F1 của checkpoint seed 3407',['negative','neutral','positive'],[{name:'F1',values:[.8481059213,.5341745531,.6783439490],fill:TEAL,valuesFormatCode:'0.0000'}],735,200,480,343,{max:1,unit:.2,labels:true});
txt(s,'79 / 333 positive thành neutral. 157 / 448 neutral thành negative.',64,567,1152,47,28,N,true);
txt(s,'Một checkpoint, N=2.149. F1 negative cao nhất trong đánh giá này.',64,626,1152,29,23,MUTED);
notes(s,'Hàng của confusion matrix là nhãn thật, cột là dự đoán. Negative đúng 1.153 trên 1.368; neutral đúng 254 trên 448; positive đúng 213 trên 333. F1 lần lượt là 0,8481, 0,5342 và 0,6783. Khi so mean C0 với B5 qua ba seeds, class weighting giảm F1 negative từ 0,8596 xuống 0,8438, tăng neutral từ 0,4754 lên 0,5410 và positive từ 0,6196 lên 0,6895. Sự đánh đổi theo lớp giúp giải thích tại sao Accuracy ít tăng nhưng Macro-F1 cải thiện. Phân biệt các số của checkpoint demo trên hình với trung bình ba seeds.','results/tables/final_per_class.csv ; results/tables/baseline_final_per_class.csv ; results/metrics/B5-balanced_seed-3407_test.json');}
// 14. Short, anchored qualitative cases with uncertainty.
{
const s=slide('Lỗi chứa cảm xúc pha trộn và thiếu ngữ cảnh','ERROR ANALYSIS');
txt(s,'Tweet 9332   negative, dự đoán positive',64,203,1152,46,29,BLUE,true);
txt(s,'“awesome... Doors close in 2 minutes ... WTH?”\nLời khen đối lập sự cố, ứng viên mỉa mai cần nhóm rà soát',64,257,1152,106,28);
txt(s,'Tweet 1537   negative, dự đoán positive',64,385,1152,46,29,BLUE,true);
txt(s,'“good try but @SouthwestAir got her here safer and sooner”\nHai hãng, hai mục tiêu cảm xúc. P0 gộp mentions thành usermarker',64,438,1152,108,28);
txt(s,'85 case đã có nhận xét của trợ lý. Chưa có hai người review độc lập.\nCase có chủ đích không ước tính tỷ lệ sarcasm của toàn dataset.',64,590,1152,65,23,MUTED);
notes(s,'Tweet 9332 dùng awesome ở đầu, nhưng WTH và sự cố gợi ý mỉa mai. Tweet 1537 khen hãng khác trong khi sentiment gán cho hãng được đề cập là negative; P0 làm mất tên hãng trong mentions. Đây là giả thuyết dựa trên văn bản, chưa có attribution hoặc ablation cleaning để xác định nguyên nhân. Trợ lý đã đọc 85 case raw/clean; nhóm chưa có hai người đánh giá độc lập. Accuracy của nhóm tweet dài cao hơn nhưng Macro-F1 thấp và negative chiếm đa số, nên chưa kết luận tweet dài dễ hơn. Giữ nhãn dataset khi tính metric, kể cả case có nhãn đáng xem lại.','results/predictions/assistant_error_review.csv ; results/predictions/manual_error_review.csv ; results/tables/error_slices.csv');}
// 15. Real demo screen and reproducible command.
{
const s=slide('Demo tải đúng checkpoint đã đánh giá','INFERENCE');
await fig(s,'F23_demo_report',55,183,785,445);
txt(s,'B5-balanced, seed 3407\nCheckpoint epoch 16\n\nNhập tweet tiếng Anh\nNhận nhãn và 3 xác suất',885,211,330,224,29,N,true);
txt(s,'Địa chỉ local\n127.0.0.1:8765\n\nSoftmax chưa calibration',885,489,330,142,25,MUTED);
notes(s,'Chuyển sang demo local, nhập câu cảm ơn như ảnh hoặc dùng một testcase đã lưu. Demo tải model và tokenizer đã đóng băng. Ba giá trị hiển thị là output softmax chưa hiệu chỉnh; giá trị 98% không đảm bảo xác suất dự đoán đúng ngoài thực tế là 98%. Có thể thử một case sai để trình bày hạn chế. Nếu runtime chưa sẵn trên máy trình chiếu, dùng ảnh chụp request thật đã kiểm tra này và nói rõ đang trình bày ảnh. Demo chưa được hosting public. Hướng dẫn khởi động có trong README.','airline_rnn/demo.py ; results/metrics/demo_verification.json ; results/figures/F23_demo.png');}
// 16. Synthesis and honest limits.
{
const s=slide('Kết quả có thể truy vết, neutral còn là hạn chế','KẾT LUẬN');
txt(s,'22 run RNN với checkpoint, history và prediction đã lưu',64,202,1152,70,33,N,true);
txt(s,'Class weighting nâng Macro-F1 trong miền đã khảo sát\nT60 thêm PAD và chi phí, không thêm nội dung văn bản\nModel cuối có demo dùng cùng preprocessing và label map',64,303,1152,156,29);
txt(s,'Giới hạn của kết luận',64,494,1152,45,30,BLUE,true);
txt(s,'Một split, ba training seeds, tweet tiếng Anh trong một miền\nChưa human adjudication, calibration hoặc chạy lại trên cloud\nMở rộng cần holdout mới, giữ RNN là mô hình trọng tâm',64,553,1152,110,26,MUTED);
notes(s,'Kết quả chính gồm quy trình có thể chạy lại và số liệu thực đo. Cần đặt 90% của nguồn LSTM binary và 75% của RNN ba lớp trong các điều kiện đánh giá khác nhau, đồng thời giữ đầy đủ các training seeds. Class weighting cải thiện các lớp ít mẫu trong miền đã khảo sát, với sự đánh đổi ở negative. Bước tiếp theo là nhóm rà 85 case, điền đóng góp thành viên, chạy lại trên môi trường thứ hai và thiết kế holdout mới nếu khảo sát thêm. LSTM, GRU và BERT chỉ thuộc một phần mở rộng riêng nếu được thực hiện. Xin cảm ơn và sẵn sàng trao đổi.','reports/final/BaoCao_Airline_RNN.docx ; README.md ; reports/project_status.md');}
await fs.writeFile(path.join(ROOT,'reports/final/presentation_content.json'),JSON.stringify(texts,null,2));
const candidate=path.join(TMP,'candidate.pptx');await (await PresentationFile.exportPptx(P)).save(candidate);
const final=path.join(ROOT,'reports/final',process.env.DECK_NAME??'ThuyetTrinh_Airline_RNN_generated.pptx');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...([...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],nativeChartTargetApplication:'portable',materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(TMP,path.basename(final)+'.validation.json')});
console.log(JSON.stringify({final,slides:slides.length,font:FONT,result}));
await fs.mkdir(path.join(TMP,'slides-preview'),{recursive:true});
for(let i=0;i<slides.length;i++){const preview=await P.export({slide:slides[i],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'slides-preview',`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));}
console.log('Rendered 16 slide previews.');
